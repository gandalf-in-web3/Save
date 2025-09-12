ray start --head --port=6379 --resources='{"slot": 1}'

# export CUDA_VISIBLE_DEVICES=1

ray start --address='192.168.4.23:6379' --resources='{"slot": 1}' --num-gpus=1

ray stop
