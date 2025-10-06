#!/bin/bash
# Worker script for performing gradient computation on worker nodes
set -e

# Parse arguments
MODE=$1          # compute or apply_weights
ITER=$2          # iteration number
NODE_RANK=$3     # 1 or 2
WEIGHTS_FILE=$4  # Weights file passed from master node

# Configuration
SCRIPT_DIR="$HOME/dpo/compo"  # This need to be modified
MODEL_PATH="$HOME/Mistral-Instruct-7B-DPO_clean" # This also need to be modified
LOG_DIR="$SCRIPT_DIR/logs"
GRAD_DIR="$SCRIPT_DIR/gradients"
BATCH_DIR="$SCRIPT_DIR/noisy_batches"
WORKER_WEIGHTS_DIR="$SCRIPT_DIR/worker_weights"
OUTPUT_PATH="$SCRIPT_DIR/output"

# Create directories
mkdir -p $LOG_DIR $GRAD_DIR $BATCH_DIR $WORKER_WEIGHTS_DIR

# Activate environment (if using virtualenv)
source ~/dpo/env/bin/activate

export NCCL_P2P_DISABLE=1

echo "Worker node $NODE_RANK starting mode $MODE for iteration $ITER with weights file $WEIGHTS_FILE"

if [ "$MODE" == "compute" ]; then
    # Compute gradients mode
    ITER_DIR="$GRAD_DIR/iter_$ITER"
    ITER_BATCH_DIR="$BATCH_DIR/iter_$ITER"
    mkdir -p $ITER_DIR $ITER_BATCH_DIR
    
    # Verify weights file exists
    if [ ! -f "$WEIGHTS_FILE" ]; then
        echo "ERROR: Weights file $WEIGHTS_FILE not found for compute"
        exit 1
    fi
    
    python $SCRIPT_DIR/compute_gradients.py \
      --model_path=$MODEL_PATH \
      --output_path=$OUTPUT_PATH \
      --grad_dir=$GRAD_DIR \
      --noisy_batch_dir=$BATCH_DIR \
      --weights_file=$WEIGHTS_FILE \
      --node_rank=$NODE_RANK \
      --mode=compute \
      --iteration=$ITER \
      --num_perturbation=1800 \
      2>&1 | tee $LOG_DIR/worker${NODE_RANK}_compute_iter$ITER.log
      
    # Check that gradient files were created
    if [ ! -f "$ITER_DIR/node_grad_vector_${NODE_RANK}.npy" ] || [ ! -f "$ITER_DIR/node_grad_meta_${NODE_RANK}.json" ]; then
        echo "ERROR: Failed to create gradient files for node $NODE_RANK"
        exit 1
    fi
    
    echo "Worker node $NODE_RANK successfully completed gradient computation for iteration $ITER"
    
elif [ "$MODE" == "apply_weights" ]; then
    # Apply weights mode
    if [ ! -f "$WEIGHTS_FILE" ]; then
        echo "ERROR: Weights file $WEIGHTS_FILE not found for apply_weights"
        exit 1
    fi
    
    # Always use base model - we only update the lm_head weights
    python $SCRIPT_DIR/compute_gradients.py \
      --model_path=$MODEL_PATH \
      --output_path=$OUTPUT_PATH \
      --weights_file=$WEIGHTS_FILE \
      --node_rank=$NODE_RANK \
      --mode=apply_weights \
      --iteration=$ITER \
      2>&1 | tee $LOG_DIR/worker${NODE_RANK}_apply_weights_iter$ITER.log
      
    echo "Worker node $NODE_RANK successfully applied weights for iteration $ITER"
    
else
    echo "ERROR: Unknown mode $MODE"
    exit 1
fi