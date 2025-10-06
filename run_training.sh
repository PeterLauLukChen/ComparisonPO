#!/bin/bash
# Main training script for master node (node 0)
set -e

# Configuration - need to be set
SCRIPT_DIR="$HOME/dpo/compo" # Your working directory
WORKER1_IP="10.121.24.23"    # The internal IP address for worker node 1
WORKER2_IP="10.121.24.24"    # The internal IP address for worker node 2
MODEL_PATH="/home/User/dpo/compo/Mistral-Instruct-7B-DPO_clean" # Path to your model
source ~/dpo/env/bin/activate # Change to your env
# Set the first n noisy sample to run
VALID_ITERATIONS=(0 1 2 3 4 5 6 7 8 9) # First 9 noisy preference pair 
User="YOUR WORKER NODES USERNAME FOR SSH" # Input your node user name

# Configurations - default directory for master node processing 
LOG_DIR="$SCRIPT_DIR/logs"
GRAD_DIR="$SCRIPT_DIR/gradients"
BATCH_DIR="$SCRIPT_DIR/noisy_batches"
TEMP_WEIGHTS_DIR="$SCRIPT_DIR/temp_weights"
WORKER_WEIGHTS_DIR="$SCRIPT_DIR/worker_weights"
OUTPUT_PATH="$SCRIPT_DIR/output"

# Create directories
mkdir -p $LOG_DIR $GRAD_DIR $BATCH_DIR $TEMP_WEIGHTS_DIR $WORKER_WEIGHTS_DIR

EXTRACT_SCRIPT="$SCRIPT_DIR/extract_lm_head.py"
if [ ! -f "$EXTRACT_SCRIPT" ]; then
    cat > "$EXTRACT_SCRIPT" << 'EOF'
#!/usr/bin/env python
import argparse
import torch
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser(description='Extract lm_head weights from model')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--output_file', type=str, required=True, help='Path to save weights')
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path)
torch.save(model.lm_head.weight.data.cpu(), args.output_file)
print(f"Saved lm_head weights to {args.output_file}")
EOF
    chmod +x "$EXTRACT_SCRIPT"
    echo "Created extract_lm_head.py script"
fi

# Initialize temp weights file from base model if it doesn't exist
TEMP_WEIGHTS="$TEMP_WEIGHTS_DIR/current_lm_head.pt"
if [ ! -f "$TEMP_WEIGHTS" ]; then
    echo "Initializing temp weights from base model..."
    python $EXTRACT_SCRIPT --model_path=$MODEL_PATH --output_file=$TEMP_WEIGHTS
fi

# Activate environment (if using virtualenv)
export NCCL_P2P_DISABLE=1

# Copy script to worker nodes and ensure it's executable
echo "Copying scripts to worker nodes..."
for worker in "$WORKER1_IP" "$WORKER2_IP"; do
    ssh User@$worker "mkdir -p $SCRIPT_DIR/logs $SCRIPT_DIR/gradients $SCRIPT_DIR/noisy_batches $SCRIPT_DIR/worker_weights"
    scp $SCRIPT_DIR/compute_gradients.py User@$worker:$SCRIPT_DIR/
    scp $SCRIPT_DIR/run_worker.sh User@$worker:$SCRIPT_DIR/
    scp $EXTRACT_SCRIPT User@$worker:$SCRIPT_DIR/
    ssh User@$worker "chmod +x $SCRIPT_DIR/run_worker.sh"
    ssh User@$worker "chmod +x $SCRIPT_DIR/extract_lm_head.py"
done

# Reset offset to start fresh
OFFSET=0
echo "Starting with dataset offset: $OFFSET"

# Get command line arguments
NUM_ITERATIONS=${1:-100}
START_ITER=${2:-0}

# Run for specified number of iterations
for ITER in $(seq $START_ITER $((NUM_ITERATIONS-1))); do
    echo "==========================================="
    echo "Processing iteration $ITER"
    echo "==========================================="
    
    # Check if this iteration is in the valid list
    VALID=0
    for VALID_ITER in "${VALID_ITERATIONS[@]}"; do
        if [ "$ITER" -eq "$VALID_ITER" ]; then
            VALID=1
            break
        fi
    done
    
    if [ "$VALID" -eq "0" ]; then
        echo "Iteration $ITER not in valid list - just advancing dataset index"
        
        # Create minimal directories needed
        ITER_BATCH_DIR="$BATCH_DIR/iter_$ITER"
        mkdir -p $ITER_BATCH_DIR
        
        # Just find a noisy pair to advance the dataset pointer
        python $SCRIPT_DIR/compute_gradients.py \
            --model_path=$MODEL_PATH \
            --output_path=$OUTPUT_PATH \
            --grad_dir=$GRAD_DIR \
            --noisy_batch_dir=$BATCH_DIR \
            --weights_file=$TEMP_WEIGHTS \
            --node_rank=0 \
            --mode=find_pairs \
            --iteration=$ITER \
            --dataset_offset=$OFFSET \
            --batch_size=1 \
            --noise_threshold=5.0 \
            2>&1 | tee $LOG_DIR/advance_index_iter$ITER.log
        
        # Update dataset offset for next iteration
        if [ -f "$GRAD_DIR/master_offset.txt" ]; then
            OFFSET=$(cat "$GRAD_DIR/master_offset.txt")
            echo "Advanced dataset offset to: $OFFSET"
        fi
        
        # Skip the rest of processing
        continue
    fi

    echo "Iteration $ITER is in valid list - performing full update"
    
    # Create iteration directories
    ITER_DIR="$GRAD_DIR/iter_$ITER"
    ITER_BATCH_DIR="$BATCH_DIR/iter_$ITER"
    mkdir -p $ITER_DIR $ITER_BATCH_DIR
    
    # Clear any existing files from previous runs of this iteration
    rm -f $ITER_DIR/*.npy $ITER_DIR/*.json $ITER_BATCH_DIR/*.pt $ITER_BATCH_DIR/*.json
    
    # Step 1: Find noisy pairs on master node
    echo "Finding noisy pairs for iteration $ITER..."

    python $SCRIPT_DIR/compute_gradients.py \
        --model_path=$MODEL_PATH \
        --output_path=$OUTPUT_PATH \
        --grad_dir=$GRAD_DIR \
        --noisy_batch_dir=$BATCH_DIR \
        --weights_file=$TEMP_WEIGHTS \
        --node_rank=0 \
        --mode=find_pairs \
        --iteration=$ITER \
        --dataset_offset=$OFFSET \
        --batch_size=1 \ # Current version of ComPO performed best at batchsize = 1, this should not be changed
        --noise_threshold=3.0 \ # Used to filter noisy pair via log probablity 
        2>&1 | tee $LOG_DIR/find_pairs_iter$ITER.log
    
    # Check if pairs were found successfully
    if [ ! -f "$ITER_BATCH_DIR/batch_meta.json" ]; then
        echo "ERROR: Failed to find noisy pairs for iteration $ITER. Exiting."
        exit 1
    fi
    
    # Update dataset offset for next iteration
    if [ -f "$GRAD_DIR/master_offset.txt" ]; then
        OFFSET=$(cat "$GRAD_DIR/master_offset.txt")
        echo "Updated dataset offset to: $OFFSET for next iteration"
    fi
    
    # Step 2: Copy noisy batches and current weights to worker nodes
    echo "Copying noisy batches and weights to worker nodes..."
    # Create worker-specific weight files
    WORKER1_WEIGHTS="$WORKER_WEIGHTS_DIR/worker1_weights.pt"
    WORKER2_WEIGHTS="$WORKER_WEIGHTS_DIR/worker2_weights.pt"
    cp "$TEMP_WEIGHTS" "$WORKER1_WEIGHTS"
    cp "$TEMP_WEIGHTS" "$WORKER2_WEIGHTS"
    
    # Copy to worker nodes
    for worker in "$WORKER1_IP" "$WORKER2_IP"; do
        ssh User@$worker "mkdir -p $ITER_BATCH_DIR $SCRIPT_DIR/worker_weights"
        scp $ITER_BATCH_DIR/* User@$worker:$ITER_BATCH_DIR/
    done
    scp $WORKER1_WEIGHTS User@$WORKER1_IP:$SCRIPT_DIR/worker_weights/
    scp $WORKER2_WEIGHTS User@$WORKER2_IP:$SCRIPT_DIR/worker_weights/
    
    # Step 3: Start workers to compute gradients
    echo "Starting worker nodes for gradient computation..."
    ssh User@$WORKER1_IP "cd $SCRIPT_DIR && ./run_worker.sh compute $ITER 1 $SCRIPT_DIR/worker_weights/worker1_weights.pt" &
    WORKER1_PID=$!
    
    ssh User@$WORKER2_IP "cd $SCRIPT_DIR && ./run_worker.sh compute $ITER 2 $SCRIPT_DIR/worker_weights/worker2_weights.pt" &
    WORKER2_PID=$!
    
    # Step 4: Start master computation
    echo "Starting master node computation for iteration $ITER"
    python $SCRIPT_DIR/compute_gradients.py \
        --model_path=$MODEL_PATH \
        --output_path=$OUTPUT_PATH \
        --grad_dir=$GRAD_DIR \
        --noisy_batch_dir=$BATCH_DIR \
        --weights_file=$TEMP_WEIGHTS \
        --node_rank=0 \
        --mode=compute \
        --iteration=$ITER \
        --num_perturbation=1800 \
        2>&1 | tee $LOG_DIR/master_compute_iter$ITER.log
    
    # Step 5: Wait for workers to finish
    echo "Waiting for worker nodes to complete gradient computation..."
    wait $WORKER1_PID
    WORKER1_STATUS=$?
    wait $WORKER2_PID
    WORKER2_STATUS=$?
    
    if [ $WORKER1_STATUS -ne 0 ]; then
        echo "WARNING: Worker 1 exited with non-zero status: $WORKER1_STATUS"
    fi
    
    if [ $WORKER2_STATUS -ne 0 ]; then
        echo "WARNING: Worker 2 exited with non-zero status: $WORKER2_STATUS"
    fi
    
    # Check that gradient files exist from all nodes
    MISSING_FILES=0
    for node in 0 1 2; do
        if [ ! -f "$ITER_DIR/node_grad_vector_${node}.npy" ] || [ ! -f "$ITER_DIR/node_grad_meta_${node}.json" ]; then
            echo "WARNING: Missing gradient files for node $node"
            MISSING_FILES=1
        fi
    done
    
    if [ $MISSING_FILES -eq 1 ]; then
        echo "WARNING: Some gradient files are missing. Attempting to retrieve from worker nodes..."
        
        # Copy from worker 1
        scp User@$WORKER1_IP:$ITER_DIR/node_grad_vector_1.npy $ITER_DIR/ 2>/dev/null || true
        scp User@$WORKER1_IP:$ITER_DIR/node_grad_meta_1.json $ITER_DIR/ 2>/dev/null || true
        
        # Copy from worker 2
        scp User@$WORKER2_IP:$ITER_DIR/node_grad_vector_2.npy $ITER_DIR/ 2>/dev/null || true
        scp User@$WORKER2_IP:$ITER_DIR/node_grad_meta_2.json $ITER_DIR/ 2>/dev/null || true
    fi
    
    # Step 6: Aggregate gradients and update model
    echo "Aggregating gradients and updating model..."
    ITER_WEIGHTS="$TEMP_WEIGHTS_DIR/lm_head_weights_iter$ITER.pt"

    python $SCRIPT_DIR/compute_gradients.py \
        --model_path=$MODEL_PATH \
        --output_path=$OUTPUT_PATH \
        --grad_dir=$GRAD_DIR \
        --noisy_batch_dir=$BATCH_DIR \
        --weights_file=$TEMP_WEIGHTS \
        --node_rank=0 \
        --mode=aggregate \
        --iteration=$ITER \
        --num_perturbation=1800 \
        --temp_weights_file=$ITER_WEIGHTS \
        2>&1 | tee $LOG_DIR/master_aggregate_iter$ITER.log

    # If weights were successfully updated, copy to temp weights file
    if [ -f "$ITER_WEIGHTS" ]; then
        cp "$ITER_WEIGHTS" "$TEMP_WEIGHTS"
        echo "Updated temp weights file for next iteration"
        
        # Step 7: Transfer just the weights to worker nodes (not the full model)
        echo "Transferring updated weights to worker nodes..."
        
        # Create worker-specific weight files
        cp "$ITER_WEIGHTS" "$WORKER1_WEIGHTS"
        cp "$ITER_WEIGHTS" "$WORKER2_WEIGHTS"
        
        # For worker 1 (use rank 1)
        ssh User@$WORKER1_IP "mkdir -p $SCRIPT_DIR/worker_weights"
        scp $WORKER1_WEIGHTS User@$WORKER1_IP:$SCRIPT_DIR/worker_weights/
        ssh User@$WORKER1_IP "cd $SCRIPT_DIR && ./run_worker.sh apply_weights $ITER 1 $SCRIPT_DIR/worker_weights/worker1_weights.pt" &
        
        # For worker 2 (use rank 2)
        ssh User@$WORKER2_IP "mkdir -p $SCRIPT_DIR/worker_weights"
        scp $WORKER2_WEIGHTS User@$WORKER2_IP:$SCRIPT_DIR/worker_weights/
        ssh User@$WORKER2_IP "cd $SCRIPT_DIR && export NCCL_P2P_DISABLE=1 && ./run_worker.sh apply_weights $ITER 2 $SCRIPT_DIR/worker_weights/worker2_weights.pt" &

        wait
        
        echo "Iteration $ITER completed successfully"
    else
        echo "ERROR: No weights file generated. Model update failed."
        exit 1
    fi

    echo "Cleaning up iteration files..."
    # Keep the current iteration weights but clean up other files
    find "$GRAD_DIR/iter_$ITER" -type f -name "*.npy" -delete
    find "$GRAD_DIR/iter_$ITER" -type f -name "*.json" -delete
    find "$BATCH_DIR/iter_$ITER" -type f -name "*.pt" -delete
    find "$BATCH_DIR/iter_$ITER" -type f -name "*.json" -delete

    # Also clean up on worker nodes
    for worker in "$WORKER1_IP" "$WORKER2_IP"; do
        ssh User@$worker "find $GRAD_DIR/iter_$ITER -type f -name \"*.npy\" -delete" 2>/dev/null || true
        ssh User@$worker "find $GRAD_DIR/iter_$ITER -type f -name \"*.json\" -delete" 2>/dev/null || true
        ssh User@$worker "find $BATCH_DIR/iter_$ITER -type f -name \"*.pt\" -delete" 2>/dev/null || true
        ssh User@$worker "find $BATCH_DIR/iter_$ITER -type f -name \"*.json\" -delete" 2>/dev/null || true
    done

done

echo "Multi-node training completed successfully!"