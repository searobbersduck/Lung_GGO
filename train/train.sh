# CUDA_VISIBLE_DEVICES=0,1 python train.py --arch=final --block_size 128
# CUDA_VISIBLE_DEVICES=0,1 python train.py --arch=final --block_size 64
CUDA_VISIBLE_DEVICES=0,1 python train.py --arch=final --block_size 32

# CUDA_VISIBLE_DEVICES=0,1 python train.py --arch=middle --block_size 128
# CUDA_VISIBLE_DEVICES=0,1 python train.py --arch=middle --block_size 64
CUDA_VISIBLE_DEVICES=0,1 python train.py --arch=middle --block_size 32