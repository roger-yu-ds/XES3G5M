export PYTHONPATH=$(pwd)
export CUDA_LAUNCH_BLOCKING=1
MODEL=sakt
# MODEL=sakt_with_additive_pre_embeddings
PRETRAINED_MODEL_DIR=artifacts/sakt/sakt_causal_w_overlap_0_1_2_3_2025-05-04_10-12-25

python src/main.py \
    --model=$MODEL \
    --pretrained_model_dir=$PRETRAINED_MODEL_DIR
