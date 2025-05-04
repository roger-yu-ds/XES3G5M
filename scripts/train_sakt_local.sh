export PYTHONPATH=$(pwd)
export CUDA_LAUNCH_BLOCKING=1
# MODEL=sakt
MODEL=sakt
RUN_NAME=sakt_causal_w_pre_embeddings_overlap
PRE_EMBEDDING_LIST=(question)
OVERLAP_SIZE=100

python src/main.py \
    --model=$MODEL \
    --pre_embedding_list ${PRE_EMBEDDING_LIST[@]} \
    --run_name=$RUN_NAME \
    --overlap_size=$OVERLAP_SIZE