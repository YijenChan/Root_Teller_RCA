set -x
WORKSPACE="${THINKFL_WORKSPACE:-$PWD}"

# Optional cleanup and directory creation should be performed under WORKSPACE.

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "."}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --save_steps 1 \
    --lr_warmup_ratio 0.01 \
    --pretrain "$WORKSPACE/checkpoint/Llama-3_2-3B-sft3" \
    --save_path "$WORKSPACE/checkpoint/Llama-3_2-3B-grpo-mix2" \
    --ckpt_path "$WORKSPACE/checkpoint/Llama-3_2-3B-grpo-mix2-ckpt" \
    --save_hf_ckpt \
    --load_checkpoint \
    --save_grader_result_path "$WORKSPACE/training_output/Llama-3_2-3B-grpo-mix2/grader_result.jsonl" \
    --remote_rm_url "$WORKSPACE/reward_func.py" \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 128 \
    --num_episodes 5 \
    --max_epochs 1 \
    --prompt_max_len 3000 \
    --generate_max_len 5000 \
    --advantage_estimator group_norm \
    --n_samples_per_prompt 4 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 1e-6 \
    --init_kl_coef 0.0 \
    --temperature 0.5 \
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
