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
from triton.testing import do_bench
import itertools


torch.npu.set_device(0)
# ========== 全局变量和常量 ==========
DEVICE = "npu"
TEST_DATA_DIR = "/home/coder/fa-test-batch/test_data"
RESULT_DIR = "./test_results"
RESULT_DIR = "./test_results_batch"
os.makedirs(RESULT_DIR, exist_ok=True)

import os
os.environ["TRITON_BENCH_METHOD"] = "npu" # 设置为 NPU 测试方法
os.environ["TRITON_PRINT_AUTOTUNING"] = "1" # 打印自动调优信息

test_results = []  # 全局结果存储
valid_fields = ["B", "N1", "S1", "D", "causal", "dtype", "BM", "BN", "From", "Testcase Name", "sparse mode"]
dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32, 
             'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32,
             'torch.bfloat16': torch.bfloat16, 'torch.float16': torch.float16, 'torch.float32': torch.float32}

# D 泛化列表, GPU 仅支持 D 为 2 的幂次方
D_FANHUA_LIST = [64, 128]

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


def get_autotune_config():
    configs = []

    BM_list = [64, 128, 256]
    BN_list = [64, 128, 256, 512]

    multibuffer_list = [True]  # [True, False]
    unit_flag_list = [True]  # [True, False]
    limit_auto_multi_buffer_only_for_local_buffer_list = [False]  # [True, False]
    limit_auto_multi_buffer_of_local_buffer_list = ["no-l0c"]  # ["no-limit", "no-l0c"]

    # These knobs are tuned only when limit_auto_multi_buffer_only_for_local_buffer=False
    set_workspace_multibuffer_list = [2, 4]  # [2, 4]
    enable_hivm_auto_cv_balance_list = [True]  # [True, False]
    tile_mix_vector_loop_num_list = [2, 4]  # [2, 4]
    tile_mix_cube_loop_num_list = [2, 4]  # [2, 4]

    # BM_list = [256]
    # BN_list = [256]

    # set_workspace_multibuffer_list = [4]  # [2, 4]
    # enable_hivm_auto_cv_balance_list = [True]  # [True, False]
    # tile_mix_vector_loop_num_list = [4]  # [2, 4]
    # tile_mix_cube_loop_num_list = [2]  # [2, 4]


    for (
        BM,
        BN,
        multibuffer,
        unit_flag,
        limit_auto_multi_buffer_only_for_local_buffer,
        limit_auto_multi_buffer_of_local_buffer,
    ) in itertools.product(
        BM_list,
        BN_list,
        multibuffer_list,
        unit_flag_list,
        limit_auto_multi_buffer_only_for_local_buffer_list,
        limit_auto_multi_buffer_of_local_buffer_list,
    ):

        if limit_auto_multi_buffer_only_for_local_buffer:
            # Keep defaults when tuning doesn't make sense
            configs.append(
                triton.Config(
                    {"BLOCK_M": BM, "BLOCK_N": BN},
                    multibuffer=multibuffer,
                    unit_flag=unit_flag,
                    limit_auto_multi_buffer_only_for_local_buffer=limit_auto_multi_buffer_only_for_local_buffer,
                    limit_auto_multi_buffer_of_local_buffer=limit_auto_multi_buffer_of_local_buffer,
                )
            )
        else:
            # Fully expand tuning space
            for (
                set_workspace_multibuffer,
                enable_hivm_auto_cv_balance,
                tile_mix_vector_loop,
                tile_mix_cube_loop,
            ) in itertools.product(
                set_workspace_multibuffer_list,
                enable_hivm_auto_cv_balance_list,
                tile_mix_vector_loop_num_list,
                tile_mix_cube_loop_num_list,
            ):
                configs.append(
                    triton.Config(
                        {"BLOCK_M": BM, "BLOCK_N": BN},
                        multibuffer=multibuffer,
                        unit_flag=unit_flag,
                        limit_auto_multi_buffer_only_for_local_buffer=limit_auto_multi_buffer_only_for_local_buffer,
                        limit_auto_multi_buffer_of_local_buffer=limit_auto_multi_buffer_of_local_buffer,
                        set_workspace_multibuffer=set_workspace_multibuffer,
                        enable_hivm_auto_cv_balance=enable_hivm_auto_cv_balance,
                        tile_mix_vector_loop=tile_mix_vector_loop,
                        tile_mix_cube_loop=tile_mix_cube_loop,
                    )
                )

    return configs


@triton.autotune(
    configs=get_autotune_config(),
    key=['Z', 'H', 'N_CTX', 'HEAD_DIM'],  # 加入 shape 相关的关键参数
)
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
              ):
    # ???, why
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H
    NUM_BLOCKS_PER_CORE = (NUM_BLOCKS+ 19) // 20

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
        assert HEAD_DIM_K in {64, 128, 256}  # 注释用于泛化测试 HEAD_DIM_K

        o = torch.empty_like(q)

        # stage = 3
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        # if is_hip():
        #     waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        #     extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}
        num_cores = 20
        acc = torch.zeros((q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K), dtype=torch.float32, device=q.device)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[(num_cores,)](
            q, k, v, M, o, sm_scale, acc, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1], N_CTX=q.shape[2],  # why varidic??
            HEAD_DIM=HEAD_DIM_K,  # 64
            STAGE=stage,
            # 以下参数用于autotune
            # BLOCK_M=128,
            # BLOCK_N=512,
            # multibuffer=True, # autotune config, 控制开double buffer
            # unit_flag=True, # autotune config, cube搬出的一个优化项
            # limit_auto_multi_buffer_only_for_local_buffer=False, # autotune config, 是否开启cube和vector的并行，false表示开启
            # set_workspace_multibuffer=4, # autotune config, 表示同时cube和vector有几个并行，【2,4】，仅limit_auto_multi_buffer_only_for_local_buffer=False 时生效
            # enable_hivm_auto_cv_balance=True,
            # tile_mix_vector_loop=4,
            # tile_mix_cube_loop=4,
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

    # 【NPU】当前 kernel 功能只支持 S1 整除 BN 和 BM 的情况，非整除的过滤掉（会导致精度问题）
    if 'S1' in extracted_data.columns and 'BN' in extracted_data.columns:
        extracted_data = extracted_data[extracted_data['S1'] % extracted_data['BN'] == 0]
    if 'S1' in extracted_data.columns and 'BM' in extracted_data.columns:
        extracted_data = extracted_data[extracted_data['S1'] % extracted_data['BM'] == 0]
    # 过滤掉 D >= 256 的案例
    if 'D' in extracted_data.columns:
        extracted_data = extracted_data[extracted_data['D'] < 256]

    # 重置索引
    extracted_data = extracted_data.reset_index(drop=True)
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
        other_paths = {
            "cv融合": os.path.join(TEST_DATA_DIR, "cv_cases.xlsx"),
            "模型规格": os.path.join(TEST_DATA_DIR, "model_cases.xlsx"),
        }
        paths = {
            "step64": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_step64_case_d64_Result.xls"),
            "step64+7": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_step64+7_d64_Result.xls"),
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

        # # 01.模型规格数据 + cv融合数据 (测试案例共 18 个)
        # test_data_01 = extract_test_case_data(other_paths, extract_map, new_field, filter_data)
        # print(f"\n>>>> 其他测试文件共生成 {len(test_data_01)} 个测试案例。")
        # # 02.提取测试数据，进行抽样，抽样比例128，进行D泛化 (测试案例共 32 个)
        # test_data_02 = extract_test_case_data(paths, extract_map, new_field, filter_data, sampling=True, sampling_rows=128,
        #                                       insert_row={"D": D_FANHUA_LIST})
        # print(f">>>> 泛化测试文件共生成 {len(test_data_02)} 个测试案例。")
       
        # test_data = pd.concat([test_data_01, test_data_02], ignore_index=True).reset_index(drop=True)

        # test_cases = [row[valid_fields].to_dict() for _, row in test_data.iterrows()]
        # print(f">>>> 总共生成 {len(test_cases)} 个测试案例。") # 67

        # # （全量）确保只对 test_case 参数化一次
        # # metafunc.parametrize("test_case", test_cases, ids=[f"{case['From']}_{case['Testcase Name']}" for case in test_cases])

        # # （分批）对 test_case 参数化
        # # 选择当前批次的测试案例
        # start_index = metafunc.config.getoption("start_index")
        # batch_size = metafunc.config.getoption("batch_size")
        # end_index = min(start_index + batch_size, len(test_cases))
        # batch_cases = test_cases[start_index:end_index]
        # print(f">>> Running test batch: {start_index} to {end_index-1} ({len(batch_cases)} cases)")
        # metafunc.parametrize("test_case", batch_cases, 
        #                     ids=[f"{case['From']}_{case['Testcase Name']}" for case in batch_cases])


        # 非测试文件的测试案例
        test_cases = [
            # [4, 32, 128, 128, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_1", 0],
            # [4, 32, 64, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_2", 0],
            # [1, 2, 1024, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_3", 0],
            # [4, 32, 1024, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_4", 0],
            [4, 32, 2048, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_5", 0],
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
    q = (torch.empty((B, N1, S1, D), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    k = (torch.empty((B, N1, S1, D), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    v = (torch.empty((B, N1, S1, D), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))

    sm_scale = 0.5
    try:
        def profiling_forward_fn():
            attention(q, k, v, causal, sm_scale, BM, BN)

        # 性能测试
        kernel_avg_time = do_bench(profiling_forward_fn, rep=10)
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
            "causal": str(causal),
            "result": "Success",
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
            "causal": str(causal),
            "result": "Error",
            "Error Message": str(e),
        })
        print(f"Test case [{From}-{test_name}] failed with exception: {e}")
        pytest.fail(f"Test failed with exception [{From}-{test_name}]: {e}")
    finally:
        # 显式删除张量，释放计算图
        del q, k, v
        # 强制Python垃圾回收
        import gc
        gc.collect()
        # 额外延迟确保NPU完全重置
        time.sleep(1)


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
