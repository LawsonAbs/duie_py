CUDA_VISIBLE_DEVICES=0 python  /home/lawson/program/DuIE_py/run_1.py \
###
 # @Author: LawsonAbs
 # @Date: 2021-04-29 15:03:53
 # @LastEditTime: 2021-04-29 23:17:54
 # @FilePath: /DuIE_py/shell/run1.sh
### 
                        --do_train \
                        --n_gpu 1 \
                        --seed 42 \
                        --data_path /home/lawson/program/DuIE_py/data \
                        --max_seq_length 256 \
                        --batch_size 16 \
                        --num_train_epochs 10 \
                        --output_dir /home/lawson/program/DuIE_py/checkpoints \
                        --train_data_path /home/lawson/program/DuIE_py/data/train_data.json \
                        --dev_data_path /home/lawson/program/DuIE_py/data/dev_data.json \
                        --init_checkpoint /home/lawson/program/DuIE_py/checkpoints/model_subject_53530_bert_f1=0.787523.pdparams