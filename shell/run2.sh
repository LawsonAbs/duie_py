CUDA_VISIBLE_DEVICES=1 python /home/lawson/program/DuIE_py/run_2.py \
                                        --do_train \
                                        --n_gpu 1 \
                                        --seed 42 \
                                        --data_path /home/lawson/program/DuIE_py/data \
                                        --max_seq_length 256 \
                                        --batch_size 6 \
                                        --eval_batch_size 1 \
                                        --num_train_epochs 6 \
                                        --output_dir /home/lawson/program/DuIE_py/checkpoints \
                                        --train_data_path /home/lawson/program/DuIE_py/data/train_data.json \
                                        --dev_data_path /home/lawson/program/DuIE_py/data/dev_data_5000.json
                                        #--init_checkpoint /home/lawson/program/DuIE_py/checkpoints/model_object_42824_bert_f1=0.6979139572387244.pdparams