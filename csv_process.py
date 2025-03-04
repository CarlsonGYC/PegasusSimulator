import csv
import re

# 这个正则用来匹配 (0, 0, z, 1) 这一段，捕获其中的z。
# 注意这里的模式：\(\s*0\s*,\s*0\s*,\s*(.+?),\s*1\s*\)
pattern = re.compile(r"\(\s*0\s*,\s*0\s*,\s*(.+?),\s*1\s*\)")

input_csv_file = "./transform_log.csv"        # 原始 CSV 文件
output_csv_file = "./processed_data.csv"      # 处理后的 CSV 文件

# 以读写模式打开文件
with open(input_csv_file, "r", newline="", encoding="utf-8") as f_in, \
     open(output_csv_file, "w", newline="", encoding="utf-8") as f_out:
    
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    
    # 读一行表头（如有需要，可以保留或跳过）
    header = next(reader, None)
    
    # 也可以为新的 CSV 写一个表头，比如 ["time", "z_val"]：
    writer.writerow(["time", "z_val"])
    
    # 逐行读取
    for row in reader:
        # row[0] 是时间
        # row[1] 是包含了世界变换矩阵的字符串
        time_str = row[0]
        transform_str = row[1]
        
        # 用正则匹配提取 z 值
        match = pattern.search(transform_str)
        if match:
            z_val_str = match.group(1)  # 得到正则捕获的 z 字段
            z_val = float(z_val_str)    # 转成浮点数
            # 写入新的 CSV
            writer.writerow([time_str, z_val])
            
            # 同时在控制台打印
            print(f"time={time_str}, z={z_val}")
        else:
            # 如果匹配失败，可以根据需要处理
            print(f"time={time_str}, 未匹配到 z 值！")

print(f"数据已处理完毕，输出文件：{output_csv_file}")

