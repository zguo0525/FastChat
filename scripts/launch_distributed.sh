#!/bin/bash
export WORLD_SIZE=1
export NUM_GPU=8
export BSZ=16
export ACC_STEPS=1

export TIMESTAMP=$( date +%Y-%m-%d_%H-%M-%S )
export MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
export EXP_NAME=${MODEL##*/}_${DATA_NAME}_BSZ${BSZ}_ACC${ACC_STEPS}
export OMP_NUM_THREADS=64

# mkdir -p ${OUTPUT_DIR}
# mkdir -p ${LOG_DIR}

for i in {1..1}
do
bsub -q alt_7d -K -M 1024G -gpu "num=$NUM_GPU/task:mode=exclusive_process" -n $WORLD_SIZE  \
    -R "select[infiniband && a100_80gb] rusage[mem=1024G]" \
    -o distributed-${TIMESTAMP}.log \
    blaunch.sh bash ./scripts/train_moe.sh $WORLD_SIZE $TIMESTAMP $NUM_GPU $ACC_STEPS $EXP_NAME
done