"""
=========== FA Triton Kernel 泛化性测试 ===========
已知 测试 表格: 
  FlashAttentionScore_step64_case_d64_Result.xls 
  FlashAttentionScore_step64+7_case_d64_Result.xls

>> 每一行代表一个测试用例，包含以下字段(共78个)及示例内容：
Group: FlashAttentionScore
Testcase Name: FlashAttentionScore_BNSD_{num} / FlashAttentionScore_BSH_{num}
Enable: disable/onlypref
Level: level0
Network Type: fanhua
B: 1                    #
N1: 24
N2: 3
S1: 64
S2: 64
D: 64
Dtype: bf16
sparse mode: 0
pre tockens: 65536
next tockens: 65536
Layout: BNSD / BSH
PSE: None
pse type: None
Atten mask Dtype: None
Atten mask Shape: None
Padding Mask: None
keep prob: 1
Expect out pricision: 
Expect out err max: 
Expect out err sum: 
Expect out eb: 
Expect dp pricision: 
Actual out pricision: 0
Actual out err max: 0
Actual out err sum: 0
Actual out eb: 0
Actual dp pricision: 0
Actual dp err max: 0
Actual dp err sum: 0
Actual dp eb: 0
Actual dk pricision: 0
Actual dk err max: 0
Actual dk err sum: 0
Actual dk eb: 0
Actual dv pricision: 0
Actual dv err max: 0
Actual dv err sum: 0
Actual dv eb: 0
Actual Memory: 0
Actual kernel time forward: 0.0316
Actual kernel time backward: 0.0979
Actual e2e time forward: 0
Actual e2e time backward: 0
Precision result: Fail
Rmse result: Pass
Rme result: Pass
EB result: 
Performance result: Fail
Memory result: Fail
running status: PASS
Actual kernel time forward transpose: 0.0694
Actual kernel time backward transpose: 0.5471
Actual kernel time forward pad: 0
Actual kernel time backward pad: 0.0000
Actual kernel time forward slice: 0
Actual kernel time backward slice: 0.0000
Actual kernel time forward gpu:
Actual kernel time backward gpu:
BNSD+transpose: 0.1010
>>
任务描述：基于表格数据驱动 test_op_fwd, 完成泛化性测试（精度+性能）。
精度测试：直接 NPU 侧对比 ref_out 和 tri_out 的结果。
性能测试：与 GPU 侧对比，计算 kernel_time。
最终目的：泛化性验证 Triton 的 FA前向算子 的精度和性能。
=========== FA Triton Kernel 泛化性测试 ===========
"""

import pytest
import torch
# import torch_npu
import triton
import triton.language as tl
import os
import re
import time
import pandas as pd
import subprocess
import multiprocessing
from typing import Dict, Tuple, Optional, Any, Union, List


# ========== 全局变量和常量 ==========
DEVICE = "cuda"
TEST_DATA_DIR = "./test_data"
RESULT_DIR = "./test_results"
os.makedirs(RESULT_DIR, exist_ok=True)

import os
os.environ["TRITON_BENCH_METHOD"] = "gpu" # 设置为 GPU 测试方法
os.environ["TRITON_PRINT_AUTOTUNING"] = "1" # 打印自动调优信息

test_results = []  # 全局结果存储
valid_fields = ["B", "N1", "S1", "D", "causal", "dtype", "BM", "BN", "From", "Testcase Name", "sparse mode"]
dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32, 
             'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32,
             'torch.bfloat16': torch.bfloat16, 'torch.float16': torch.float16, 'torch.float32': torch.float32}

# D 泛化列表
D_FANHUA_LIST = [64, 72, 80, 88, 96, 128, 256]

# ========== Triton Kernel 社区实现（保持不变）:https://github.com/triton-lang/triton/blob/v3.2.0/python/tutorials/06-fused-attention.py ==========
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            # p = p.to(tl.float16)
            p = p.to(v.dtype) # 需修改成 v.dtype 支持 bf16
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# def get_autotune_config():
#     return [
#         triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_stages=1, num_warps=1),
#         triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_stages=1, num_warps=1),

#         triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=1, num_warps=1),
#         triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_stages=1, num_warps=1),

#         triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=1, num_warps=1),

#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_stages=1, num_warps=1),
#         # triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=1, num_warps=1),
#         # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=1, num_warps=1),
#     ]

# values = {"has_exception": False}


# def _post_hook(*args, exception):
#     if exception is not None:
#         print(f">> Triton kernel exception: {exception}")
#         values["has_exception"] = True


# @triton.autotune(
#     configs=get_autotune_config(),
#     key=['Z', 'H', 'N_CTX', 'HEAD_DIM'],  # 加入 shape 相关的关键参数
#     post_hook=_post_hook,
# )

# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 64 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=['Z', 'H', 'N_CTX', 'HEAD_DIM'])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        # assert HEAD_DIM_K in {16, 32, 64, 128, 256} # 注释该信息
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
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
            # "prof01": os.path.join(TEST_DATA_DIR, "prof_case_274.xlsx"),
            # "prof01": os.path.join(TEST_DATA_DIR, "prof_case_all.xlsx"),
            # "prof01": os.path.join(TEST_DATA_DIR, "prof_case_test_gpu.xlsx"),
            # "step64": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_test.xls"),
            # "step64": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_step64_case_d64_Result.xls"),
            # "step64+7": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_step64+7_d64_Result.xls"),
            # "extract": os.path.join(RESULT_DIR, "extract_test_case_prof.xlsx"),
            "retest": os.path.join(TEST_DATA_DIR, "space_retest_gpu.xlsx"),
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
        

        # 提取测试数据
        # test_data = extract_test_case_data(paths, extract_map, new_field, filter_data, sampling=False, sampling_rows=128,
        #                                    insert_row={"D": D_FANHUA_LIST})
        # test_data = extract_test_case_data(paths, extract_map, new_field, filter_data)

        # test_cases = [row[valid_fields].to_dict() for _, row in test_data.iterrows()]
        # # 确保只对 test_case 参数化一次
        # metafunc.parametrize("test_case", test_cases, ids=[f"{case['From']}_{case['Testcase Name']}" for case in test_cases])

        # 非测试文件的测试案例
        test_cases = [
            # [1, 128, 8192, 192, False, torch.bfloat16, 64, 64, "模型规格", "DeepSeekV2_0001", 0],
            [1, 14, 111800, 128, False, torch.bfloat16, 64, 64, "模型规格", "MFU_0001", 0],
            [1, 14, 251300, 128, False, torch.bfloat16, 64, 64, "模型规格", "MFU_0002", 0],
            [24, 5, 9216, 64, False, torch.float16, 64, 64, "模型规格", "XingHuoTuWenSD_RealCase_0001", 0],
            [24, 10, 2304, 64, False, torch.float16, 64, 64, "模型规格", "XingHuoTuWenSD_RealCase_0003", 0],
            [2, 8, 4096, 128, False, torch.bfloat16, 64, 64, "模型规格", "LLaMa_RealCase_0007", 0],
            [1, 12, 4096, 128, False, torch.bfloat16, 64, 64, "模型规格", "PanGuZhiZi_RealCase_0001", 0],
            [1, 12, 4096, 128, False, torch.float16, 64, 64, "模型规格", "PanGuZhiZi_RealCase_0002", 0],
            [1, 4, 4096, 256, False, torch.float16, 64, 64, "模型规格", "PanGuZhiZi_RealCase_0003", 0],
            [1, 8, 8192, 128, False, torch.bfloat16, 64, 64, "模型规格", "TongYiQianWen_RealCase_0001", 0],
            [1, 10, 32768, 128, False, torch.bfloat16, 64, 64, "模型规格", "X1_long_seq_173", 0],
            [1, 5, 32768, 128, False, torch.bfloat16, 64, 64, "模型规格", "X1_long_seq_174", 0],
            [2, 10, 32768, 128, False, torch.bfloat16, 64, 64, "模型规格", "X1_long_seq_175", 0],
            [2, 5, 32768, 128, False, torch.bfloat16, 64, 64, "模型规格", "X1_long_seq_176", 0],
            [4, 32, 128, 128, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_1", 0],
            [4, 32, 64, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_2", 0],
            [1, 2, 1024, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_3", 0],
            [4, 32, 1024, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_4", 0],
            [4, 32, 2048, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_5", 0],
            [4, 32, 4096, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_6", 0],
            [4, 32, 8192, 64, False, torch.float16, 32, 32, "cv融合", "FlashAttentionScore_BNSD_7", 0],
            [4, 32, 16384, 64, False, torch.float16, 32, 32, "cv融合", "FlashAttentionScore_BNSD_8", 0],

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
    q = (torch.empty((B, N1, S1, D), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    k = (torch.empty((B, N1, S1, D), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    v = (torch.empty((B, N1, S1, D), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))

    sm_scale = 0.5
    try:
        #  ==== triton kernel 精度测试 ====
        # tri_out = attention(q, k, v, causal, sm_scale, BM, BN)

        # M = torch.tril(torch.ones((S1, S1), device=DEVICE))
        # p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

        # if causal:
        #     p[:, :, M==0] = float('-inf')
        # p = torch.softmax(p.float(), dim=-1).half().to(v.dtype)

        # ref_out = torch.matmul(p, v)

        # atol, rtol = precision_atol_rtol(dtype)         # 误差分析
        # errors = compute_errors(ref_out, tri_out)
        # passed = torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)
        #  ==== triton kernel 精度测试 ====

        def profiling_forward_fn():
            with torch.no_grad():
                attention(q, k, v, causal, sm_scale)

        # ==== triton kernel 性能测试 ====
        kernel_avg_time = do_bench(profiling_forward_fn, keep_res=False, rep=20)
        print(f">> Kernel average time: {kernel_avg_time}")

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
            # "Precision result": "Pass" if passed else "Fail",
            # **{f"Actual out {k}": v for k, v in errors.items()},
            "Actual kernel time forward": kernel_avg_time,
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
            "Error Message": str(e),
        })
        print(f"Test case [{From}-{test_name}] failed with exception: {e}")
        pytest.fail(f"Test failed with exception [{From}-{test_name}]: {e}")
    finally:
        # 显式删除张量，释放计算图
        del q, k, v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()


def collect_single(base_dir: str, key: str = None) -> float:
    if not os.path.exists(base_dir):
        return float('inf')

    import pandas as pd
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file != 'op_statistic.csv':
                continue
            target_file = os.path.join(root, file)
            df = pd.read_csv(target_file)
            if key is not None:
                key_rows = df[df['OP Type'].str.startswith(key, na=False)]
                if not key_rows.empty:
                    return key_rows['Avg Time(us)'].values[0]
                return float('inf')
            else:
                # default: read the first row except header
                return df.loc[0, 'Avg Time(us)']

    return float('inf')

def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean", keep_res=False):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all" Default is "mean".    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]
    import torch

    enable_bench_npu = os.getenv("TRITON_BENCH_METHOD", 'default').lower() in ('npu')
    enable_bench_gpu = os.getenv("TRITON_BENCH_METHOD", 'default').lower() in ('gpu')
    if enable_bench_npu:
        avg_times = do_bench_npu(fn, warmup=max(5, warmup), active=max(10, rep), keep_res=keep_res)
        print(f"Average time on NPU: {avg_times:.2f} us")
        return _summarize_statistics(torch.tensor([avg_times], dtype=torch.float), quantiles, return_mode)
    elif enable_bench_gpu:
        avg_times = do_bench_gpu(fn, warmup=max(5, warmup), active=max(10, rep))
        return _summarize_statistics(torch.tensor([avg_times], dtype=torch.float), quantiles, return_mode)


def _summarize_statistics(times, quantiles, return_mode):
    import torch
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(torch, return_mode)(times).item()


def do_bench_npu(fn, warmup=5, active=30, prof_dir=None, keep_res=False):
    import torch
    import torch_npu
    from datetime import datetime, timezone

    # warmup kernel
    fn()
    torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    skip_first = 10
    wait = 0
    repeat = 1
    total = skip_first + (wait + warmup + active) * repeat

    if prof_dir is not None:
        torch_path = prof_dir
    else:
        process = multiprocessing.current_process()
        pid = process.pid
        process_name = process.name
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        torch_path = os.path.join("profile_results", f"prof_{timestamp}_{process_name}-{pid}")
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total):
            fn()
            prof.step()
            torch.npu.synchronize()
    time = collect_single(torch_path)
    del prof

    if not keep_res:
        import shutil
        if os.path.exists(torch_path):
            shutil.rmtree(torch_path)

    return time


def do_bench_gpu(fn, warmup=5, active=30):
    from datetime import datetime, timezone

    # warmup kernel
    fn()
    torch.cuda.synchronize()

    skip_first = 10
    wait = 0
    repeat = 1
    total = skip_first + (wait + warmup + active) * repeat
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
    ) as prof:
        torch.cuda.synchronize()
        for i in range(total):
            fn()
            prof.step()
        torch.cuda.synchronize()

    times = parse_prof(prof)
    del prof

    return times


def parse_prof(prof):
    event_list = prof.events()
    parsed_times = []

    for evt in event_list:
        if evt.device_type == torch.profiler.DeviceType.CUDA:
            parsed_times.append(evt.device_time_total)
    
    return parsed_times
