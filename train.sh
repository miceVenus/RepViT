NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model repvit_m1_5 --data-path E:\\DataSet\\val --dist-eval
