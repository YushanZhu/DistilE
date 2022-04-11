DATA_DIR=dataset

MODEL_NAME=ComplEx
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=distil_KGELitModel
MAX_EPOCHS=5000
EMB_DIM=64
TEACHER_DIM=64
LOSS=DistilE_Loss
ADV_TEMP=1.0
TRAIN_BS=1024
EVAL_BS=16
NUM_NEG=1024
MARGIN=10.0
LR=1e-4
REGULARIZATION=5e-6
CHECK_PER_EPOCH=10
EARLY_STOP=100
NUM_WORKERS=16
GPU=1

WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$GPU python -u main_dist.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --loss $LOSS \
    --adv_temp $ADV_TEMP \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LR \
    --regularization $REGULARIZATION \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --teacher_dim $TEACHER_DIM \
    --save_config \
    --use_weight \
    --early_stop_patience $EARLY_STOP\






