export PYTHONPATH=$(pwd)
export CUDA_LAUNCH_BLOCKING=1
MODEL=sakt_with_additive_pre_embeddings
PRETRAINED_MODEL_DIR=artifacts/sakt/sakt_with_additive_question_pre_embeddings_0_1_2_3_2025-04-29_22-37-13
PRE_EMBEDDING_LIST=(question)

python src/main.py \
    --model=$MODEL \
    --pre_embedding_list ${PRE_EMBEDDING_LIST[@]} \
    --pretrained_model_dir=$PRETRAINED_MODEL_DIR