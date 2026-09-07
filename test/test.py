# test/04_path.py

from pathlib import Path

# 创建 Path 对象
path = Path(r"F:\Pycharm_Agent\KnowLedge\doc\hak180产品安全手册.pdf")

# 常用属性
print(path.name)    # "万用表的使用.pdf" - 完整文件名
print(path.stem)       # "万用表的使用" - 不含扩展名的文件名
print(path.suffix)     # ".pdf" - 扩展名
print(path.parent)     # Path("D:/doc") - 父目录

# 常用方法
print(path.exists())  # True/False - 文件是否存在
print(path.is_file())  # True/False - 是否为文件
print(path.is_dir())   # True/False - 是否为目录

# 路径拼接（使用 / 运算符）
output_path = path.parent / "output" / "result.md"
print(output_path)