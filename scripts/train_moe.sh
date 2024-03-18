export PYTHONFAULTHANDLER=1
source ./scripts/ccc_nccl.sh

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export RANK=$((LSF_PM_XTASKID - 1))
export MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
export MASTER_PORT=54967
echo "EXP_NAME" $EXP_NAME
echo "Distributed training:"
echo MASTER_ADDR $MASTER_ADDR
echo MASTER_PORT $MASTER_PORT
# echo RANK $RANK
export OMP_NUM_THREADS=64

# Start the GPU monitor in the background
(while :; do nvidia-smi; sleep 300; done) &

# Train the model
MKL_SERVICE_FORCE_INTEL=1

torchrun --nproc_per_node=$NUM_GPU \
    --nnodes=$WORLD_SIZE:$WORLD_SIZE \
    --rdzv_id=${EXP_NAME} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --max_restarts=2 \
    fastchat/train/train_mem.py \
    --model_name_or_path /dccstor/mit_fm/zguo0525/MOE/checkpoints/moduleformer-8b-stage2-hf/iter_0264000 \
    --data_path data/modified_openhermes2_5.json \
    --bf16 True \
    --output_dir output_moe \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${BSZ} \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps ${ACC_STEPS} \
    --evaluation_strategy "steps" \
    --eval_steps 4000 \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'ModuleFormerBlock' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True

