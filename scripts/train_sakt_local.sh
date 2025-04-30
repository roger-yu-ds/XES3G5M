export PYTHONPATH=$(pwd)
export CUDA_LAUNCH_BLOCKING=1
# MODEL=sakt
MODEL=sakt_with_additive_pre_embeddings
RUN_NAME=sakt_with_additive_question_pre_embeddings
PRE_EMBEDDING_LIST=(question)
python src/main.py \
    --model=$MODEL \
    --pre_embedding_list ${PRE_EMBEDDING_LIST[@]} \
    --learning_curve