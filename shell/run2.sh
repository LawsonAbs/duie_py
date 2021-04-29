CUDA_VISIBLE_DEVICES=2 python /home/lawson/program/DuIE_py/run_2.py \
                                        --do_train \
                                        --n_gpu 1 \
                                        --seed 42 \
                                        --data_path /home/lawson/program/DuIE_py/data \
                                        --max_seq_length 128 \
                                        --batch_size 4 \
                                        --eval_batch_size 1 \
                                        --num_train_epochs 12 \
                                        --output_dir /home/lawson/program/DuIE_py/checkpoints \
                                        --train_data_path /home/lawson/program/DuIE_py/data/train_data.json \
                                        --dev_data_path /home/lawson/program/DuIE_py/data/dev_data.json
                                        #--init_checkpoint /home/lawson/program/DuIE_py/model_object/model_object_556706_bert.pdparams