set -x
WORKSPACE="${THINKFL_WORKSPACE:-$PWD}"

rm -rf "$WORKSPACE/checkpoint/Llama-3-8B-grpo-sft5"
rm -rf "$WORKSPACE/checkpoint/Llama-3-8B-grpo-sft5-ckpt"
rm -rf "$WORKSPACE/training_output/Llama-3-8B-grpo-sft5"
mkdir -p "$WORKSPACE/checkpoint/Llama-3-8B-grpo-sft5"
mkdir -p "$WORKSPACE/checkpoint/Llama-3-8B-grpo-sft5-ckpt"
mkdir -p "$WORKSPACE/training_output/Llama-3-8B-grpo-sft5"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "."}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --save_steps 1 \
    --lr_warmup_ratio 0.01 \
    --pretrain "$WORKSPACE/checkpoint/Llama-3-8B-sft4" \
    --save_path "$WORKSPACE/checkpoint/Llama-3-8B-grpo-sft5" \
    --ckpt_path "$WORKSPACE/checkpoint/Llama-3-8B-grpo-sft5-ckpt" \
    --save_hf_ckpt \
    --load_checkpoint \
    --save_grader_result_path "$WORKSPACE/training_output/Llama-3-8B-grpo-sft5/grader_result.jsonl" \
    --remote_rm_url "$WORKSPACE/reward_func.py" \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 128 \
    --num_episodes 2 \
    --max_epochs 1 \
    --prompt_max_len 3000 \
    --generate_max_len 5000 \
    --advantage_estimator group_norm \
    --n_samples_per_prompt 16 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --init_kl_coef 1e-3 \
    --gamma 1.0 \
    --use_kl_loss \
    --kl_estimator k3 \
    --temperature 0.3 \
    --prompt_data "$WORKSPACE/dataset_mix" \
    --input_key context_messages \
    --apply_chat_template \
    --max_samples 1000 \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --vllm_sync_backend nccl \
    --gradient_checkpointing \
    --enforce_eager \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --use_wandb True \
    --wandb_host "${WANDB_HOST:-https://api.wandb.ai}" \
    --wandb_api_key "$WANDB_API_KEY"

# ray start --head --num-gpus 8
# ray start --head --num-gpus 2
# ray start --head --node-ip-address $MASTER_ADDR --num-gpus 2
# ray start --address=$MASTER_ADDR:6379 --num-gpus 2
