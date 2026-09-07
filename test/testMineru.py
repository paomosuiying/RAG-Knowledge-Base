import os
import subprocess

from tenacity import wait_chain

env =os.environ.copy()
env["MINERU_MODEL_SOURCE"]="local"

#cmd
cmd="mineru -p F:\Pycharm_Agent\KnowLedge\doc\hak180产品安全手册.pdf -o F:/output --backend pipeline"

#子进程
proc = subprocess.Popen(
    args= cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    errors="replace",
    text=True,
    encoding="utf-8",
    bufsize=1,
    env=env
)

#获取日志信息
for line in proc.stdout:
    print(f"执行Mineru产生的日志：{line}")

wait_code = proc.wait()\

if wait_code == 0:
    print("执行成功")