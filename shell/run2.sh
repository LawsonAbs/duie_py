CUDA_VISIBLE_DEVICES=1 python run_2.py --do_train --n_gpu 1 --seed 42 --data_path ./data --max_seq_length 128 --batch_size 1 --num_train_epochs 12 --output_dir ./checkpoints 
#--init_checkpoint ./model_object/model_object_214120.pdparams
