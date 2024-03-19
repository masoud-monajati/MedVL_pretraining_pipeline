CUDA_VISIBLE_DEVICES=0,2,3,4,6 python -m torch.distributed.run --nproc_per_node=5 pretrain.py --config ./configs/pretrain.yaml --output_dir /local1/monajati/med/checkpoints/pretrain
