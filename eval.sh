#!/bin/bash
set -e

EXPORTED="/data/mithilesh/vae_paper/exported"
DATASET_CONFIG="stable_audio_tools/configs/dataset_configs/local_daps_test.json"
BATCH_SIZE=2
OUT_PATH="/data/mithilesh/vae_paper/eval_output"

echo "=========================================="
echo "Running VAE 1 baseline"
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 python evaluate_autoencoder.py \
    --model_config ${EXPORTED}/vae1_baseline_config.json \
    --ckpt_path ${EXPORTED}/vae1_baseline.ckpt \
    --dataset_config ${DATASET_CONFIG} \
    --out_path ${OUT_PATH}/vae1_baseline \
    --batch_size ${BATCH_SIZE}

echo "=========================================="
echo "Running VAE 1 powerchannel"
echo "=========================================="
CUDA_VISIBLE_DEVICES=2 python evaluate_autoencoder.py \
    --model_config ${EXPORTED}/vae1_powerchannel_config.json \
    --ckpt_path ${EXPORTED}/vae1_powerchannel.ckpt \
    --dataset_config ${DATASET_CONFIG} \
    --out_path ${OUT_PATH}/vae1_powerchannel \
    --batch_size ${BATCH_SIZE}

echo "=========================================="
echo "Running VAE 2 baseline"
echo "=========================================="
CUDA_VISIBLE_DEVICES=3 python evaluate_autoencoder.py \
    --model_config ${EXPORTED}/vae2_baseline_config.json \
    --ckpt_path ${EXPORTED}/vae2_baseline.ckpt \
    --dataset_config ${DATASET_CONFIG} \
    --out_path ${OUT_PATH}/vae2_baseline \
    --batch_size ${BATCH_SIZE}

echo "=========================================="
echo "Running VAE 2 powerchannel"
echo "=========================================="
CUDA_VISIBLE_DEVICES=4 python evaluate_autoencoder.py \
    --model_config ${EXPORTED}/vae2_powerchannel_config.json \
    --ckpt_path ${EXPORTED}/vae2_powerchannel.ckpt \
    --dataset_config ${DATASET_CONFIG} \
    --out_path ${OUT_PATH}/vae2_powerchannel \
    --batch_size ${BATCH_SIZE}

echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
