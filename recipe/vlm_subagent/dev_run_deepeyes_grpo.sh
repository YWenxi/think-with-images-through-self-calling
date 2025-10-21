#!/bin/bash

set -x

export LLM_AS_A_JUDGE_BASE="http://127.0.0.1:18901/v1"
# export WANDB_API_KEY="your wandb key"

PROJECT_NAME="VLM-Subagent-RL"
EXPERIMENT_NAME="dev-DeepEyes-GRPO"

# save experiment results
EXPERIMENT_DIR="/data2/YangWenxi/subagent-exp"

BASEDIR=/data1/YangWenxi/subagent-rl/verl
SAVE_CHECKPOINT_DIR=${EXPERIMENT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/verl_checkpoints
DATASET_TRAIN=$HOME/Workspace/subagent-rl/data/DeepEyes-Datasets-47k/data_v0.8_visual_toolbox_v2.parquet
DATASET_VAL=$HOME/Workspace/subagent-rl/data/DeepEyes-Datasets-47k/data_thinklite_reasoning_acc.parquet

REF_MODEL_PATH=$HOME/Workspace/subagent-rl/pretrained_models/Qwen/Qwen2.5-VL-3B-Instruct

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export RAY_DEBUG_POST_MORTEM=1

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=${BASEDIR}/recipe/deepeyes/configs \
    --config-name='deepeyes_multiturn_grpo' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=[${DATASET_VAL}] \
    data.train_batch_size=56 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=28 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.num_workers=7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/deepeyes/configs/image_zoom_in_tool_config.yaml \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','tensorboard'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=7 \
    trainer.nnodes=1 \
    trainer.save_freq=8 \
    trainer.test_freq=80 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=1 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
