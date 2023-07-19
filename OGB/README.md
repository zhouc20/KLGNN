Experiments on ogbg-molhiv


running examples:

CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'ogbg-molhiv' --model "IDMPNN" --k 4 --distlimitu 3 --distlimitl -1 --resolution 0.5 --name --num_layers 6 --num_layers_global 5 --num_layers_id 5 --num_layers_regression 1 --node_pool 'mean' --subgraph_pool 'add' --global_pool 'max' --cat 'add' --rate 0.05 --hid_dim 64 --batch_size 32 --lr 3e-5 --epochs 80 --full_graph --ensemble_test --sample_times 5 --ensemble_mode 'median' --no 1

CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'ogbg-molhiv' --model "IDMPNN" --k 4 --distlimitu 3 --distlimitl -1 --resolution 0.5 --name --num_layers 8 --num_layers_global 5 --num_layers_id 5 --num_layers_regression 2 --node_pool 'mean' --subgraph_pool 'add' --global_pool 'max' --cat 'add' --rate 0.05 --hid_dim 96 --batch_size 32 --lr 1e-4 --epochs 100 --full_graph --ensemble_test --sample_times 5 --ensemble_mode 'median' --no 1

