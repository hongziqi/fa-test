import pandas as pd
import os

TEST_DATA_DIR = "./test_data"

table1_path = os.path.join(TEST_DATA_DIR, "tmp_ac.xlsx")  # 第一个完整表格
table2_path = os.path.join(TEST_DATA_DIR, "tmp_ac_2.xlsx")  # 第二个重测表格

# 读取两个表格
df1 = pd.read_excel(table1_path)  # 第一个完整表格
df2 = pd.read_excel(table2_path)  # 第二个重测表格

# 定义索引列
keys = ["From", "Testcase Name", "B", "N1", "S1", "D"]

# 需要被更新的字段
update_cols = [
    "BM", "BN", "causal", "result", "Precision result",
    "Actual out err max", "Actual out err sum", "Actual out err mean",
    "Actual out rmse", "Error Message"
]

# 设置索引便于查找
df2_indexed = df2.set_index(keys)

# 遍历df1，按顺序更新
for idx, row in df1.iterrows():
    key = tuple(row[k] for k in keys)
    if key in df2_indexed.index:
        for col in update_cols:
            if col in df2_indexed.columns:
                df1.at[idx, col] = df2_indexed.at[key, col]

# 保存结果
merge_path = os.path.join(TEST_DATA_DIR, "merged_result.xlsx")
df1.to_excel(merge_path, index=False, na_rep="nan")
print(f"Merged table saved to {merge_path}")
