Experiements of IDMPNN and ID-Transformer on zinc-12k

# IDMPNN baseline
CUDA_VISIBLE_DEVICES=0 python run_IDMPNN.py --dataset 'zinc12' --k 4 --distlimitu 3 --distlimitl -1 --resolution 0.5 --name 'cluster_baseline' --num_layers 8 --num_layers_global 5 --num_layers_id 5 --num_layers_regression 4 --node_pool 'mean' --subgraph_pool 'add' --global_pool 'max' --cat 'hadamard_product' --rate 0.1 --hid_dim 64 --batch_size 32 --lr 1e-3 --cos_lr --epochs 1000 --full_graph --no 1

# IDMPNN + RWSE
CUDA_VISIBLE_DEVICES=0 python run_IDMPNN.py --dataset 'zinc12' --k 4 --distlimitu 3 --distlimitl -1 --resolution 0.5 --name 'cluster_RWSE' --num_layers 8 --num_layers_global 5 --num_layers_id 5 --num_layers_regression 4 --node_pool 'mean' --subgraph_pool 'add' --global_pool 'max' --cat 'add' --rate 0.1 --hid_dim 64 --rw_steps 20 --se_dim 28 --se_type 'mlp' --batch_size 32 --lr 1e-3 --cos_lr --epochs 1000 --full_graph --no 1

# IDTransformer running example
CUDA_VISIBLE_DEVICES=0 python run_IDTransformer.py --dataset 'zinc12' --k 4 --distlimitu 3 --distlimitl -1 --resolution 0.5 --name 'cluster_GPS_baseline' --transformer --num_head 8 --num_layers 0 --num_layers_global 10 --num_layers_id 5 --num_layers_regression 0 --node_pool 'mean' --subgraph_pool 'max' --global_pool 'add' --cat 'add' --rate 0.1 --hid_dim 80 --batch_size 32 --lr 1e-3 --cos_lr --epochs 1000 --full_graph --local_MPNN --central_encoding --attn_bias --rw_steps 20  --se_dim 32 --se_type 'mlp' --no 1
