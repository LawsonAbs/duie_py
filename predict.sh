set -eux

export BATCH_SIZE=32
export CKPT=./checkpoints/model_64224.pdparams
export DATASET_FILE=./data/dev_data_1.json # 使用dev_data.json 是为了可视化最后结果的输出

CUDA_VISIBLE_DEVICES=1 python run_duie.py \
                   --do_predict \
                   --init_checkpoint $CKPT \
                   --predict_data_file $DATASET_FILE \
                   --max_seq_length 512 \
                   --batch_size $BATCH_SIZE

