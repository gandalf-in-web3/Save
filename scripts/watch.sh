# 用于监控系统运行状态的命令
# 监控进程
top

# 监控内存
watch -n 1 -t free -h

# 监控显卡
watch -n 1 -t nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv

# 监控IO
iostat -xz -m -y 1

# 监控某个进程的IO, KB_rd/s代表从磁盘读取
pidstat -d 1 -p $pid
