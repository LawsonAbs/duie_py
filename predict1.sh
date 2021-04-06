CUDA_VISIBLE_DEVICES=1 python predict1.py --do_train --data_path ./data --max_seq_length 128 --batch_size 32 --num_train_epochs 12 --output_dir ./checkpoints
