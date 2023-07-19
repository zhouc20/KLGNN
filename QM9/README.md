Implementation of ID-MPNN and ID-PPGN on QM9

% QM9 --task in [0, 1, ..., 11]
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'QM9' --task 0 --model 'MPNN' --k 4 --distlimitu 3 --distlimitl -1 --resolution 0.5 --num_layers 8 --num_layers_global 5 --num_layers_id 5 --num_layers_regression 2 --node_pool 'mean' --subgraph_pool 'add' --global_pool 'max' --cat 'add' --rate 0.05 --hid_dim 64 --batch_size 32 --epochs 400 --full_graph --ensemble_test --sample_times 5 --ensemble_mode 'median' --no 1

% use position input
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'QM9' --task 0 --model 'MPNN' --k 4 --distlimitu 3 --distlimitl -1 --resolution 0.5 --num_layers 8 --num_layers_global 5 --num_layers_id 5 --num_layers_regression 1 --node_pool 'mean' --subgraph_pool 'add' --global_pool 'max' --cat 'add' --rate 0.05 --hid_dim 64 --batch_size 32 --epochs 400 --full_graph --ensemble_test --sample_times 5 --ensemble_mode 'median' --use_pos --no 1

% IDPPGN
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'QM9' --task 0 --model 'PPGN' --k 4 --distlimitu 3 --distlimitl -1 --resolution 0.5 --num_layers 3 --num_layers_global 5 --num_layers_id 3 --num_layers_regression 1 --node_pool 'mean' --subgraph_pool 'add' --global_pool 'add' --cat 'add' --rate 0.05 --hid_dim 48 --batch_size 16 --epochs 400 --full_graph --ensemble_test --sample_times 5 --ensemble_mode 'median' --use_pos --no 1
