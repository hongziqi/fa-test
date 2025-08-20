#!/bin/bash

# 配置参数
TOTAL_CASES=224  # 总测试案例数，根据实际情况调整
BATCH_SIZE=4   # 每批运行的测试案例数量
SCRIPT_NAME="test_fa_fwd_npu_prof_tune.py"  # 您的测试脚本名称
RESULT_DIR="test_results_batch"  # 结果保存目录

# 创建结果目录
mkdir -p $RESULT_DIR

# 分批运行测试
batch_count=0
for ((start=0; start<TOTAL_CASES; start+=BATCH_SIZE)); do
    batch_count=$((batch_count+1))
    end=$((start+BATCH_SIZE-1))
    if [ $end -ge $TOTAL_CASES ]; then
        end=$((TOTAL_CASES-1))
    fi
    
    echo "=============================================="
    echo "Running test batch #$batch_count: Cases $start to $end"
    echo "=============================================="
    
    # 运行当前批次的测试
    pytest $SCRIPT_NAME \
        --start-index=$start \
        --batch-size=$BATCH_SIZE \
        -sv \
        --tb=short \
        > "$RESULT_DIR/batch_${start}_to_${end}.log" 2>&1
    
    # 检查上一个命令是否成功
    if [ $? -eq 0 ]; then
        echo "Batch $start to $end completed successfully"
    else
        echo "Batch $start to $end failed, check log for details"
    fi
    
    # 可选：在批次之间添加延迟，确保NPU完全重置
    sleep 5
done

echo "All test batches completed. Results saved in $RESULT_DIR"