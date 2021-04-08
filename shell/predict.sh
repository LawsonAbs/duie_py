
CUDA_VISIBLE_DEVICES=0 python /home/lawson/program/DuIE_py/predict.py \
                        --do_predict  \
                        --n_gpu 1  \
                        --seed 42  \
                        --data_path \
                        /home/lawson/program/DuIE_py/data \
                        --max_seq_length 128 \
                        --batch_size 64 \