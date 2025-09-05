import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import os
import re
import time
import pandas as pd
import subprocess
import multiprocessing
from typing import Dict, Tuple, Optional, Any, Union, List


torch.npu.set_device(0)
# ========== 全局变量和常量 ==========
DEVICE = "npu"
TEST_DATA_DIR = "/home/coder/fa-test/test_data"
RESULT_DIR = "./test_results"
os.makedirs(RESULT_DIR, exist_ok=True)

import os
os.environ["TRITON_BENCH_METHOD"] = "npu" # 设置为 NPU 测试方法
os.environ["TRITON_PRINT_AUTOTUNING"] = "1" # 打印自动调优信息

test_results = []  # 全局结果存储
valid_fields = ["B", "N1", "S1", "D", "causal", "dtype", "BM", "BN", "From", "Testcase Name", "sparse mode"]
dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32, 
             'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32,
             'torch.bfloat16': torch.bfloat16, 'torch.float16': torch.float16, 'torch.float32': torch.float32}

# D 泛化列表
D_FANHUA_LIST = [64, 72, 80, 88, 96, 128, 256]

# ========== Triton Kernel 实现 0904（保持不变） ==========
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

## this version support HEAD_DIM > 128 and golden baseline is ascendC
@triton.jit
def _attn_fwd_inner(acc_ptr, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    # causal = true
    # stage = 1
    # 因果注意力，顾名思义，它在计算时会限制信息的流动，只允许模型看到当前位置及之前的位置
    # 的信息。也就是说，当前位置的输出只能依赖于该位置及其之前的输入，而不能访问当前位置
    # 之后的信息。因果注意力保证了数据的顺序性，避免了“未来信息”的泄露。
    # 但是后面的逻辑也会触发
    if STAGE == 1:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    # k 之前的版本，随路做转置的版本
    # K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    
    # 修改后不转的版本
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    row = tl.arange(0, BLOCK_M)[:, None]
    col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
    block2d_acc = row * HEAD_DIM + col_head_dim

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        
        # 修改K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        # ------------------------------

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        # p = tl.math.exp2(qk)
        p = tl.math.exp(qk)

        # [bm, head_dim] * [bn, head_dim].transpose
        if fp8_v:
            p_cast = p.to(tl.float8e5)
        else:
            # p = p.to(tl.float16)      // FIXHERE to bf16 unspported
            p_cast = p.to(k.dtype)
        v = tl.load(V_block_ptr)

        pv = tl.dot(p_cast, v)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp(m_i - m_ij)
        # alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        if HEAD_DIM < 256:
            acc_ptr = acc_ptr * alpha[:, None]
            acc_ptr = tl.dot(p_cast, v, acc_ptr)
        else:
            acc = tl.load(acc_ptr + block2d_acc)
            acc0 = tl.extract_slice(acc,(0, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
            alpha0 = tl.extract_slice(alpha, [0], [BLOCK_M // 4], [1])
            acc0 = acc0 * alpha0[:, None]
            pv0 = tl.extract_slice(pv, (0, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
            acc0 = pv0 + acc0
            acc = tl.insert_slice(acc, acc0, (0, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))

            acc1 = tl.extract_slice(acc,(BLOCK_M // 4, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
            alpha1 = tl.extract_slice(alpha, [BLOCK_M // 4], [BLOCK_M // 4], [1])
            acc1 = acc1 * alpha1[:, None]
            pv1 = tl.extract_slice(pv, (BLOCK_M // 4, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
            acc1 = pv1 + acc1
            acc = tl.insert_slice(acc, acc1, (BLOCK_M // 4, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))

            acc2 = tl.extract_slice(acc,(BLOCK_M // 2, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
            alpha2 = tl.extract_slice(alpha, [BLOCK_M // 2], [BLOCK_M // 4], [1])
            acc2 = acc2 * alpha2[:, None]
            pv2 = tl.extract_slice(pv, (BLOCK_M // 2, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
            acc2 = pv2 + acc2
            acc = tl.insert_slice(acc, acc2, (BLOCK_M // 2, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))

            acc3 = tl.extract_slice(acc,(3 * BLOCK_M // 4, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
            alpha3 = tl.extract_slice(alpha, [3 * BLOCK_M // 4], [BLOCK_M // 4], [1])
            acc3 = acc3 * alpha3[:, None]
            pv3 = tl.extract_slice(pv, (3 * BLOCK_M // 4, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
            acc3 = pv3 + acc3
            acc = tl.insert_slice(acc, acc3, (3 * BLOCK_M // 4, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))

            tl.store(acc_ptr + block2d_acc, acc)

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc_ptr, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, M, Out, sm_scale: tl.constexpr, acc, #
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,  #
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,  #
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,  #
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,  #
              Z: tl.constexpr, H: tl.constexpr, 
              N_CTX: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              NUM_BLOCKS_PER_CORE: tl.constexpr,
              NUM_BLOCKS: tl.constexpr,
              NUM_BLOCKS_M: tl.constexpr,
              ):
    # ???, why
    pid = tl.program_id(0)
    block_start = pid * NUM_BLOCKS_PER_CORE
    NUM_BLOCKS_hz = NUM_BLOCKS // NUM_BLOCKS_M
    task_m_idx = 0
    task_hz_idx = 0

    for block_idx in range(pid, NUM_BLOCKS, 20):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            # doesn't matter
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),

            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),

            # doesn't matter
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,

            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),

            offsets=(0, 0),
            # why block_n??
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # initialize offsets
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        if HEAD_DIM < 256:
            acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        else:
            acc_offset = (
                off_z.to(tl.int64) * stride_qz // stride_qm * HEAD_DIM +
                off_h.to(tl.int64) * stride_qh // stride_qm * HEAD_DIM +
                task_m_idx * BLOCK_M * HEAD_DIM
            )
            acc_ptr = acc + acc_offset
        # load scales

        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE

        if STAGE & 1:
            acc_ptr, l_i, m_i = _attn_fwd_inner(acc_ptr, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                                task_m_idx, sm_scale,  #
                                                BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                                4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                                )
        # stage 2: on-band

        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            acc_ptr, l_i, m_i = _attn_fwd_inner(acc_ptr, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                                task_m_idx, sm_scale,  #
                                                BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                                2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                                )
        # epilogue
        # m_i += tl.math.log2(l_i)
        m_i += tl.math.log(l_i)
        if HEAD_DIM < 256:
            accumulator = acc_ptr / l_i[:, None]
        else:
            row = tl.arange(0, BLOCK_M)[:, None]
            col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
            block2d_acc = row * HEAD_DIM + col_head_dim
            accumulator = tl.load(acc_ptr + block2d_acc)
            accumulator = accumulator / l_i[:, None]

        m_ptrs = M + task_hz_idx * N_CTX + offs_m

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, BM, BN):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {64, 80, 96, 128, 256}  # 注释用于泛化测试 HEAD_DIM_K

        o = torch.empty_like(q)

        # stage = 3
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        # if is_hip():
        #     waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        #     extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}
        num_cores = 20
        NUM_BLOCKS_M = triton.cdiv(q.shape[2], BM)
        NUM_BLOCKS = NUM_BLOCKS_M * q.shape[0] * q.shape[1]
        NUM_BLOCKS_PER_CORE = triton.cdiv(NUM_BLOCKS, num_cores)
        # grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        acc = torch.zeros((q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K), dtype=torch.float32, device=q.device)
        # grid = (triton.cdiv(q.shape[2], BM),1, 1)
        # (1, 2, 1024)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[(num_cores,)](
            q, k, v, M, o, sm_scale, acc, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1], N_CTX=q.shape[2],  # why varidic??
            HEAD_DIM=HEAD_DIM_K,  # 64
            BLOCK_M = BM, # 32
            BLOCK_N = BN, # 32
            STAGE=stage,
            NUM_BLOCKS_PER_CORE=NUM_BLOCKS_PER_CORE,
            NUM_BLOCKS=NUM_BLOCKS,
            NUM_BLOCKS_M=NUM_BLOCKS_M,
            **extra_kern_args)


        ctx.save_for_backward(q, k, v, o, M)
        # ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


attention = _attention.apply
# ========== Triton Kernel 实现（保持不变） ==========


def normalize_col_name(col: str) -> str:
    """
    标准化列名：去除非字母数字字符，转小写
    """
    return re.sub(r'[\s_\-]+', '', str(col).strip().lower())


def standardize_dataframe_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    根据 mapping 字典（原始列名 -> 目标字段名），标准化 DataFrame 的列名。
    :param df: 原始 DataFrame
    :param mapping: 原始列名（任意格式） -> 目标字段名（如 "Z", "H"）
    :return: 列名已标准化的 DataFrame
    """
    # 创建标准化后的列名到原始列名的映射
    normalized_to_raw = {normalize_col_name(col): col for col in df.columns}
    
    # 查找 mapping 中每个期望列名的实际列
    rename_dict = {}
    missing = []
    for raw_name, target_name in mapping.items():
        normalized_key = normalize_col_name(raw_name)
        if normalized_key in normalized_to_raw:
            actual_col = normalized_to_raw[normalized_key]
            rename_dict[actual_col] = target_name
        else:
            missing.append(raw_name)
    
    if missing:
        print(f"文件中缺失列: {missing}")
    
    # 重命名并返回
    return df.rename(columns=rename_dict)


def extract_test_case_data(
        paths: Dict[str, str],
        extract_map: Dict[str, str],
        new_field: Optional[Dict[str, Any]] = None,
        filter_data: Optional[Dict[str, Any]] = None,
        sampling: bool = False,
        sampling_rows: int = 128,
        insert_row: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
    """
    从多个 Excel 文件中提取测试用例数据。
    :param paths: 多个文件路径, 例如 {"file1": "path/to/file1", "file2": "path/to/file2"}
    :param extract_map: 提取字段映射
    :param new_field: 新字段及其值, 例如 {"new_field": "value"}
    :param filter_data: 可选的过滤条件，字典形式，键为字段名，值为期望的值
    :param sampling: 是否进行采样，默认为 True
    :param sampling_rows: 采样行数，默认为 128
    :param insert_row: 基于原来案例+需改变的字段，插入新的测试数据, 例如{"D": [64, 72, 80, 88, 96, 128, 256]}, 即每个 D 值都生成一个新的测试案例
    :param save_path: 可选的保存路径，如果提供则将结果保存为 Excel 文件
    :return: 提取的测试用例数据
    Example:
    paths = {
        "64": "FlashAttentionScore_step64_case_d64_Result.xls",
        "7": "FlashAttentionScore_step64+7_case_d64_Result.xls"
    }
    extract_map = {
        "Testcase Name": "Testcase Name",
        "Level": "Level",
        "Network Type": "Network Type",
        "B": "Z",
        "N1": "H",
        "S1": "N_CTX",
        "D": "HEAD_DIM",
        "Dtype": "dtype",
        "sparse mode": "sparse_mode",
        "Layout": "Layout",
        # 其他需要提取的字段...
    }
    new_field = {
        "BM": 32,
        "BN": 32,
        "causal": False,
    }
    """
    # 临时设置显示选项
    pd.set_option('display.max_columns', None)        # 显示所有列
    pd.set_option('display.width', None)              # 自动换行
    pd.set_option('display.max_colwidth', 50)         # 列宽足够
    dfs = []
    for key, path in paths.items():
        df = pd.read_excel(path)
        # 检查是否有From列，如果没有则添加
        if 'From' not in df.columns:
            df.insert(0, 'From', key)
        df_std = standardize_dataframe_columns(df, extract_map)
        dfs.append(df_std)
    if not dfs:
        raise ValueError("所有文件加载失败，请检查路径")

    combined_df = pd.concat(dfs, ignore_index=True).fillna("")  # 合并所有 DataFrame

    # 提取并重命名字段
    target_cols = list(extract_map.values())
    # 只保留 extract_map 中定义的目标字段
    extracted_data = combined_df[target_cols].copy()

    # 如果有新字段，添加到 DataFrame 中
    if new_field:
        for field, value in new_field.items():
            extracted_data.loc[:, field] = value
    # 映射数据类型
    if 'dtype' in extracted_data.columns:
        extracted_data.loc[:, 'dtype'] = extracted_data['dtype'].map(dtype_map)
    # 确保 From 是首列
    columns = ["From"] + [col for col in extracted_data.columns if col != "From"]
    extracted_data = extracted_data[columns]
    # 如果有过滤条件，应用过滤
    if filter_data:
        for key, value in filter_data.items():
            if key in extracted_data.columns:
                extracted_data = extracted_data[extracted_data[key] == value]
            else:
                print(f"警告: 过滤条件中的字段 '{key}' 在数据中不存在，跳过该过滤条件。")
    # 抽样
    if sampling and len(extracted_data) > sampling_rows:
       # 保留首行，然后每隔 sampling_rows 行采样一行
        idxs = [0] + [i for i in range(sampling_rows, len(extracted_data), sampling_rows)]
        extracted_data = extracted_data.iloc[idxs].reset_index(drop=True)
    # 如果有插入行，基于原来案例+需改变的字段，插入新的测试数据
    if insert_row:
        # 生成新的测试案例
        new_rows = []
        for _, row in extracted_data.iterrows():
            for field, values in insert_row.items():
                for value in values:
                    new_row = row.to_dict()
                    new_row[field] = value
                    new_rows.append(new_row)
        # 将新行添加到 DataFrame 中
        extracted_data = pd.DataFrame(new_rows)
    # 如果指定了保存路径，则保存到 Excel 文件
    if save_path:
        extracted_data.to_excel(save_path, index=False)
        print(f"提取的数据已保存到 {save_path}")
    # 展示前10行数据
    # print("Extracted test cases (head):\n", extracted_data.head(10))
    return extracted_data


def precision_atol_rtol(dtype) -> Tuple[float, float]:
    """
    根据数据类型返回精度的绝对误差和相对误差
    """
    return {
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (5e-3, 5e-3),
        torch.float32: (1e-4, 1e-4),
    }.get(dtype, (1e-4, 1e-4))


def compute_errors(ref: torch.Tensor, tri: torch.Tensor) -> Dict[str, float]:
    """
    计算多种误差指标
    """
    diff = (ref - tri).abs()
    return {
        "err max": diff.max().item(),
        "err sum": diff.sum().item(),
        "err mean": diff.mean().item(),
        "rmse": torch.sqrt((diff ** 2).mean()).item(),
    }


# 测试用例生成
def pytest_generate_tests(metafunc):
    """
    pytest hook to generate test cases dynamically
    """
    if 'test_case' in metafunc.fixturenames:
        # 生成测试用例数据
        paths = {
            # "step64_01": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_step64_case_d64_Result_01.xls"),
            # "step64_02": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_step64_case_d64_Result_02.xls"),
            # "step64+7_01": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_step64+7_d64_Result_01.xls"),
            # "step64+7_02": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_step64+7_d64_Result_02.xls"),
            "step64": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_step64_case_d64_Result.xls"),
            "step64+7": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_step64+7_d64_Result.xls"),
            # "test": os.path.join(TEST_DATA_DIR, "prof_case_test.xlsx"),
        }
        extract_map = {
            "From": "From",
            "Testcase Name": "Testcase Name",
            "B": "B",
            "N1": "N1",
            "S1": "S1",
            "D": "D",
            "Dtype": "dtype",
            "sparse mode": "sparse mode",
            "Layout": "Layout",
        }
        new_field = {
            "BM": 64,
            "BN": 64,
            "causal": False,
        }
        filter_data = {
            "Layout": "BNSD",  # 只测试 BNSD 布局(4096)
        }
        # # 提取测试数据
        # test_data = extract_test_case_data(paths, extract_map, new_field, filter_data, sampling=True, sampling_rows=32,
        #                                    insert_row={"D": D_FANHUA_LIST}, save_path=f"{RESULT_DIR}/extract_test_case_ac.xlsx")
        # # test_data = extract_test_case_data(paths, extract_map, new_field, filter_data, sampling=True, sampling_rows=32,
        # #                             insert_row={"D": D_FANHUA_LIST}, save_path=f"{RESULT_DIR}/extract_test_case_ac.xlsx")

        # test_cases = [row[valid_fields].to_dict() for _, row in test_data.iterrows()]
        # # 确保只对 test_case 参数化一次
        # metafunc.parametrize("test_case", test_cases, ids=[f"{case['From']}_{case['Testcase Name']}" for case in test_cases])

        # 非测试文件的测试案例
        test_cases = [
            # [1, 3, 53255, 128, False, torch.bfloat16, 64, 64, "step64+7", "FlashAttentionScore_BNSD_0833", 0],
            # [1, 24, 15296, 64, False, torch.bfloat16, 64, 64, "step64_01", "FlashAttentionScore_BNSD_0239", 0],
            # [1, 3, 75328, 64, False, torch.bfloat16, 64, 64, "step64_02", "FlashAttentionScore_BNSD_1177", 0],
            # [1, 3, 64000, 64000, False, torch.bfloat16, 64, 64, "step64", "FlashAttentionScore_BNSD_0153", 0],
            # [1, 24, 9792, 72, False, torch.bfloat16, 64, 64, "step64", "FlashAttentionScore_BNSD_0153", 0],
            [1, 128, 8192, 192, False, torch.bfloat16, 64, 64, "模型规格", "DeepSeekV2_0001", 0],
            # [1, 14, 111800, 128, False, torch.bfloat16, 64, 64, "模型规格", "MFU_0001", 0],
            # [1, 14, 251300, 128, False, torch.bfloat16, 64, 64, "模型规格", "MFU_0002", 0],
            # [24, 5, 9216, 64, False, torch.float16, 64, 64, "模型规格", "XingHuoTuWenSD_RealCase_0001", 0],
            # [24, 10, 2304, 64, False, torch.float16, 64, 64, "模型规格", "XingHuoTuWenSD_RealCase_0003", 0],
            # [2, 8, 4096, 128, False, torch.bfloat16, 64, 64, "模型规格", "LLaMa_RealCase_0007", 0],
            # [1, 12, 4096, 128, False, torch.bfloat16, 64, 64, "模型规格", "PanGuZhiZi_RealCase_0001", 0],
            # [1, 12, 4096, 128, False, torch.float16, 64, 64, "模型规格", "PanGuZhiZi_RealCase_0002", 0],
            # [1, 4, 4096, 256, False, torch.float16, 64, 64, "模型规格", "PanGuZhiZi_RealCase_0003", 0],
            # [1, 8, 8192, 128, False, torch.bfloat16, 64, 64, "模型规格", "TongYiQianWen_RealCase_0001", 0],
            # [1, 10, 32768, 128, False, torch.bfloat16, 64, 64, "模型规格", "X1_long_seq_173", 0],
            # [1, 5, 32768, 128, False, torch.bfloat16, 64, 64, "模型规格", "X1_long_seq_174", 0],
            # [2, 10, 32768, 128, False, torch.bfloat16, 64, 64, "模型规格", "X1_long_seq_175", 0],
            # [2, 5, 32768, 128, False, torch.bfloat16, 64, 64, "模型规格", "X1_long_seq_176", 0],
            [4, 32, 128, 128, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_1", 0],
            # [4, 32, 64, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_2", 0],
            # [1, 2, 1024, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_3", 0],
            # [4, 32, 1024, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_4", 0],
            # [4, 32, 2048, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_5", 0],
            # [4, 32, 4096, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_6", 0],
            # [4, 32, 8192, 64, False, torch.float16, 32, 32, "cv融合", "FlashAttentionScore_BNSD_7", 0], # NPU out of memory. Tried to allocate 64.00 GiB
            # [4, 32, 16384, 64, False, torch.float16, 32, 32, "cv融合", "FlashAttentionScore_BNSD_8", 0], # NPU out of memory. Tried to allocate 64.00 GiB

        ]
        metafunc.parametrize("test_case", test_cases, ids=[f"{case[8]}_{case[10]}" for case in test_cases])


def test_op_fwd(test_case:  Union[Dict[str, Any], List[Any]]):
    if isinstance(test_case, list):
        # 如果是列表，转换为字典
        test_case = {k: v for k, v in zip(valid_fields, test_case)}
    B, N1, S1, D, causal, dtype, BM, BN, From, test_name, sparse_mode = [test_case[k] for k in valid_fields]
    print(f"\n======  Running: {From}-{test_name} || B={B}, N1={N1}, S1={S1}, D={D}, causal={causal}, dtype={dtype}  ======\n")
    torch.manual_seed(20)
    # 创建输入张量 BNSD
    q = (torch.empty((B, N1, S1, D), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((B, N1, S1, D), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((B, N1, S1, D), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())

    sm_scale = 0.5
    try:
        # triton kernel
        tri_out = attention(q, k, v, causal, sm_scale, BM, BN)

        ref_out = torch_npu.npu_fusion_attention(
            q, k, v, N1,
            padding_mask=None,
            atten_mask=None,
            scale=sm_scale,
            keep_prob=1.0,
            input_layout="BNSD",
            pre_tockens=65535,
            next_tockens=65535,
            sparse_mode=0,
            )[0]

        atol, rtol = precision_atol_rtol(dtype)         # 误差分析
        errors = compute_errors(ref_out, tri_out)
        passed = torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol, equal_nan=True)

        test_results.append({
            "From": From,
            "Testcase Name": test_name,
            "B": B,
            "N1": N1,
            "S1": S1,
            "D": D,
            "Dtype": dtype,
            "sparse mode": str(sparse_mode),
            "Layout": "BNSD",
            "BM": BM,
            "BN": BN,
            "causal": str(causal),
            "result": "Success",
            "Precision result": "Pass" if passed else "Fail",
            **{f"Actual out {k}": str(v) for k, v in errors.items()},
        })
    except Exception as e:
        # 捕获异常并记录测试结果
        test_results.append({
            "From": From,
            "Testcase Name": test_name,
            "B": B,
            "N1": N1,
            "S1": S1,
            "D": D,
            "Dtype": dtype,
            "sparse mode": str(sparse_mode),
            "Layout": "BNSD",
            "BM": BM,
            "BN": BN,
            "causal": str(causal),
            "result": "Error",
            "Precision result": "Error",
            "Error Message": str(e),
        })
        print(f"Test case [{From}-{test_name}] failed with exception: {e}")
        pytest.fail(f"Test failed with exception [{From}-{test_name}]: {e}")
    finally:
        # 显式删除张量，释放计算图
        del q, k, v
