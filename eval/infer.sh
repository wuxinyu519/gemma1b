#!/bin/bash
#PBS -N final_eval
#PBS -o final_eval.out
#PBS -e final_eval.err
#PBS -l walltime=24:00:00
#PBS -q poderoso
#PBS -l select=1:ncpus=8:ngpus=1:host=gpu008
# poderoso
cd $PBS_O_WORKDIR

module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate gemma

# GPU
export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
echo "Starting unified evaluation at $(date)"
nvidia-smi


# Template1 GT :./infinite_bench/infinitebench_gpt_gt/cleaned_tags_jsonl/individual_files 
# Template 2 GT: ./infinite_bench/infinitebench_gpt_gt/gpt_groundtruth_gpt-4o 
# Template 2 changed: ./infinite_bench/infinitebench_gpt_gt/gpt_groundtruth_gpt-4o_change  

python final_infer.py \
    --checkpoint_path ../rlhf/dpo_gemma1b_own_filtered/final_model \
    --data_dir ./infinite_bench/infinitebench_gpt_gt/gpt_groundtruth_gpt-4o \
    --output_prefix ./infinite_bench/gemma1b_own_filtered_template2 \
    --batch_size 10 \
    --save_interval 50 \
    --device auto \

echo "Unified evaluation completed at $(date)"
