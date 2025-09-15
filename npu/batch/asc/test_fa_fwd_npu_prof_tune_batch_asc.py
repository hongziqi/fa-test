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

# D 泛化列表
# D_FANHUA_LIST = [64, 72, 80, 88, 96, 128, 256]
D_FANHUA_LIST = [64, 128, 256]


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

    # 【NPU】当前 kernel 功能只支持 S1 整除 BN 和 BM 的情况，非整除的过滤掉（会导致精度问题）
    if 'S1' in extracted_data.columns and 'BN' in extracted_data.columns:
        extracted_data = extracted_data[extracted_data['S1'] % extracted_data['BN'] == 0]
    if 'S1' in extracted_data.columns and 'BM' in extracted_data.columns:
        extracted_data = extracted_data[extracted_data['S1'] % extracted_data['BM'] == 0]
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

        # 01.模型规格数据 + cv融合数据 (测试案例共 19 个)
        test_data_01 = extract_test_case_data(other_paths, extract_map, new_field, filter_data)
        print(f"\n>>>> 其他测试文件共生成 {len(test_data_01)} 个测试案例。")
        # 02.提取测试数据，进行抽样，抽样比例128，进行D泛化 (测试案例共 48 个)
        test_data_02 = extract_test_case_data(paths, extract_map, new_field, filter_data, sampling=True, sampling_rows=128,
                                              insert_row={"D": D_FANHUA_LIST})
        print(f">>>> 泛化测试文件共生成 {len(test_data_02)} 个测试案例。")
       
        test_data = pd.concat([test_data_01, test_data_02], ignore_index=True).reset_index(drop=True)

        test_cases = [row[valid_fields].to_dict() for _, row in test_data.iterrows()]
        print(f">>>> 总共生成 {len(test_cases)} 个测试案例。") # 67

        # （全量）确保只对 test_case 参数化一次
        # metafunc.parametrize("test_case", test_cases, ids=[f"{case['From']}_{case['Testcase Name']}" for case in test_cases])

        # （分批）对 test_case 参数化
        # 选择当前批次的测试案例
        start_index = metafunc.config.getoption("start_index")
        batch_size = metafunc.config.getoption("batch_size")
        end_index = min(start_index + batch_size, len(test_cases))
        batch_cases = test_cases[start_index:end_index]
        print(f">>> Running test batch: {start_index} to {end_index-1} ({len(batch_cases)} cases)")
        metafunc.parametrize("test_case", batch_cases, 
                            ids=[f"{case['From']}_{case['Testcase Name']}" for case in batch_cases])


        # # 非测试文件的测试案例
        # test_cases = [
        #     # [1, 3, 53255, 128, False, torch.bfloat16, 64, 64, "step64+7", "FlashAttentionScore_BNSD_0833", 0],
        #     # [1, 24, 15296, 64, False, torch.bfloat16, 64, 64, "step64_01", "FlashAttentionScore_BNSD_0239", 0],
        #     # [1, 3, 75328, 64, False, torch.bfloat16, 64, 64, "step64_02", "FlashAttentionScore_BNSD_1177", 0],
        #     # [1, 3, 64000, 64000, False, torch.bfloat16, 64, 64, "step64", "FlashAttentionScore_BNSD_0153", 0],
        #     # [1, 24, 9792, 72, False, torch.bfloat16, 64, 64, "step64", "FlashAttentionScore_BNSD_0153", 0],
        #     # [1, 128, 8192, 192, False, torch.bfloat16, 64, 64, "模型规格", "DeepSeekV2_0001", 0], # ok
        #     # [1, 14, 111800, 128, False, torch.bfloat16, 64, 64, "模型规格", "MFU_0001", 0],
        #     # [1, 14, 251300, 128, False, torch.bfloat16, 64, 64, "模型规格", "MFU_0002", 0],
        #     # [24, 5, 9216, 64, False, torch.float16, 64, 64, "模型规格", "XingHuoTuWenSD_RealCase_0001", 0],
        #     # [24, 10, 2304, 64, False, torch.float16, 64, 64, "模型规格", "XingHuoTuWenSD_RealCase_0003", 0],
        #     # [2, 8, 4096, 128, False, torch.bfloat16, 64, 64, "模型规格", "LLaMa_RealCase_0007", 0],
        #     # [1, 12, 4096, 128, False, torch.bfloat16, 64, 64, "模型规格", "PanGuZhiZi_RealCase_0001", 0],
        #     # [1, 12, 4096, 128, False, torch.float16, 64, 64, "模型规格", "PanGuZhiZi_RealCase_0002", 0],
        #     # [1, 4, 4096, 256, False, torch.float16, 64, 64, "模型规格", "PanGuZhiZi_RealCase_0003", 0],
        #     # [1, 8, 8192, 128, False, torch.bfloat16, 64, 64, "模型规格", "TongYiQianWen_RealCase_0001", 0],
        #     # [1, 10, 32768, 128, False, torch.bfloat16, 64, 64, "模型规格", "X1_long_seq_173", 0],
        #     # [1, 5, 32768, 128, False, torch.bfloat16, 64, 64, "模型规格", "X1_long_seq_174", 0],
        #     # [2, 10, 32768, 128, False, torch.bfloat16, 64, 64, "模型规格", "X1_long_seq_175", 0],
        #     # [2, 5, 32768, 128, False, torch.bfloat16, 64, 64, "模型规格", "X1_long_seq_176", 0],
        #     # [4, 32, 128, 128, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_1", 0],
        #     # [4, 32, 64, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_2", 0],
        #     # [1, 2, 1024, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_3", 0],
        #     # [4, 32, 1024, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_4", 0],
        #     [4, 32, 2048, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_5", 0],
        #     # [4, 32, 4096, 64, False, torch.float16, 64, 64, "cv融合", "FlashAttentionScore_BNSD_6", 0],
        #     # [4, 32, 8192, 64, False, torch.float16, 32, 32, "cv融合", "FlashAttentionScore_BNSD_7", 0], # NPU out of memory. Tried to allocate 64.00 GiB
        #     # [4, 32, 16384, 64, False, torch.float16, 32, 32, "cv融合", "FlashAttentionScore_BNSD_8", 0], # NPU out of memory. Tried to allocate 64.00 GiB
        # ]
        # metafunc.parametrize("test_case", test_cases, ids=[f"{case[8]}_{case[10]}" for case in test_cases])


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
            torch_npu.npu_fusion_attention(
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

        # 性能测试
        kernel_avg_time = do_bench_npu(profiling_forward_fn, active=15)
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
            # "Precision result": "Pass" if passed else "Fail",
            # **{f"Actual out {k}": str(v) for k, v in errors.items()},
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
    skip_first = 5
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

    if not keep_res:
        import shutil
        if os.path.exists(torch_path):
            shutil.rmtree(torch_path)

    return time
