import pandas as pd
import os

TEST_DATA_DIR = "./test_data"

table1_path = os.path.join(TEST_DATA_DIR, "tmp_ac.xlsx")  # 第一个完整表格
table2_path = os.path.join(TEST_DATA_DIR, "npu_ac_retest.xlsx")  # 第二个重测表格

# 读取两个表格
df1 = pd.read_excel(table1_path)  # 第一个完整表格
df2 = pd.read_excel(table2_path)  # 第二个重测表格

# 定义索引列
keys = ["From", "Testcase Name", "B", "N1", "S1", "D"]

# 设置索引
df1_indexed = df1.set_index(keys)
df2_indexed = df2.set_index(keys)

# 用 df2 的数据更新 df1
df1_updated = df1_indexed.combine_first(df2_indexed).reset_index()

# 保存结果
df1_updated.to_excel("merged_result.xlsx", index=False)
