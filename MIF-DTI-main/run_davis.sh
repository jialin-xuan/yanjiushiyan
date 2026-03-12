#!/bin/bash

# 如果log文件夹不存在，则创建
if [ ! -d "log" ]; then
    mkdir -p log
fi

# 获取当前时间，格式为YYYY-MM-DD_HH-MM-SS
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
comment="MIF-DTI-Davis"
file_name="log/$current_time-$comment.log"

echo "Starting experiment for Davis dataset..."
echo "Log file: $file_name"

# 运行nohup命令，输出日志到log目录下，文件名为当前时间
# 注意：这里去掉了最后的 &，因为通常在脚本中运行并不需要后台运行，除非用户明确希望脚本退出后继续运行
# 但为了保持与用户意图一致（用户用了 nohup ... &），我们可以这样做：
# 如果用户希望脚本运行完退出，那就在后台运行。
# 但脚本最后有 tail -f，说明用户希望看到输出。

# 直接运行并重定向输出，同时在终端显示（如果需要）
# 或者照搬用户的逻辑：后台运行 + tail -f

# Run on GPU with optimized memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python -u main.py Davis -f 5 -g 0 >> "$file_name" 2>&1 &

pid=$!
echo "Process started with PID: $pid"

# 等待一秒确保日志文件创建
sleep 1

# 实时显示日志文件内容
# 使用 trap 捕获 Ctrl+C，停止 tail 但不停止 python 进程
trap 'echo "Stop watching log. Process $pid is still running."; exit' INT

tail -f "$file_name"
