## Substructure counting experiments

### running examples
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'dataset1' --task 'triangle'             --num_layers 2 --lr 3e-4 --epochs 100 --cos_lr  --no 1 --k_truncated 3 
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'dataset1' --task 'chordal_cycle'        --num_layers 2 --lr 3e-4 --epochs 100 --cos_lr  --no 1 
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'dataset1' --task 'tailed_triangle'      --num_layers 2 --lr 3e-4 --epochs 100 --cos_lr  --no 1 
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'dataset1' --task 'star'                 --num_layers 2 --lr 1e-4 --epochs 100 --cos_lr  --no 1 


CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'dataset2' --task 'triangle'             --num_layers 1 --lr 3e-3 --epochs 100 --cos_lr  --no 1 --k_truncated 3
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'dataset2' --task 'chordal_cycle'        --num_layers 1 --lr 3e-3 --epochs 100 --cos_lr  --no 1 
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'dataset2' --task 'tailed_triangle'      --num_layers 2 --lr 3e-3 --epochs 100 --cos_lr  --no 1 
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'dataset2' --task 'star'                 --num_layers 2 --lr 3e-3 --epochs 100 --cos_lr  --no 1
