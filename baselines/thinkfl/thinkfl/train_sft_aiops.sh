set -x

export HF_ENDPOINT=https://hf-mirror.com
export NCCL_IGNORE_DISABLED_P2P=1
WORKSPACE="${THINKFL_WORKSPACE:-$PWD}"
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 8192 \
   --dataset ./dataset_sft \
   --input_key context_messages \
   --output_key response
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --max_samples 100000 \
   --pretrain "$WORKSPACE/checkpoint/Llama-3_2-3B" \
   --save_path "$WORKSPACE/checkpoint/Llama-3_2-3B-sft" \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --apply_chat_template
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus 8 --module $training_commands
fi
