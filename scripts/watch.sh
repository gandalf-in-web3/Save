# 用于监控系统运行状态的命令

# 监控进程
top

# 监控内存
watch -n 1 -t free -h

# 监控显卡
watch -n 1 -t nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv

# 监控某个进程的IO, KB_rd/s代表从磁盘读取
pidstat -d 1 -p $pid

# 查看磁盘的挂载方式
mount | grep data

# 查看GPU之间的拓扑结构
nvidia-smi topo -m

# 查看GPU之间是否支持P2P通信
nvidia-smi topo -p2p rw

# 查看一台机器上有哪些网卡
ip -br addr

# 查看网卡的性能
ethtool ib0
