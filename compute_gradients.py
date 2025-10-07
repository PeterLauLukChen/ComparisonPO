import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import json
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference
from datasets import load_dataset
import torch.nn.functional as F
from typing import Dict, List, Union, Any, Optional, Tuple
import logging
import torch.multiprocessing as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class WeightDeltaLayer(nn.Module):
    """
    A layer that efficiently applies perturbations to weights without full copying.
    """
    def __init__(self, base_layer):
        super().__init__()
        self.base_weight = base_layer.weight.data.clone()
        self.delta = nn.Parameter(torch.zeros_like(self.base_weight))
        self.scale = 1.0
        
    def forward(self, x):
        # Apply weights = base_weights + scale * delta
        return F.linear(x, self.base_weight + self.scale * self.delta)
        
    def set_delta(self, delta_vector, scale=1.0):
        with torch.no_grad():
            self.delta.copy_(delta_vector.reshape(self.delta.shape))
            self.scale = scale
            
    def reset(self):
        with torch.no_grad():
            self.delta.zero_()
            self.scale = 1.0

def trl_DPO_setup(model, tokenizer, dataset, tokenization):
    """Set up the DPO trainer and process the dataset."""
    logger.info("Creating DPO config and trainer")
    dpo_config = DPOConfig(
        output_dir="mistral-7b-cleani",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=5.0e-7,
        max_prompt_length=512,
        max_completion_length=1024,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    if not tokenization:
        return trainer, None

    logger.info("Processing dataset")
    processed_dataset = trainer._prepare_dataset(
        dataset=dataset,  
        processing_class=tokenizer,  
        args=dpo_config, 
        dataset_name="train", 
    )
    
    return trainer, processed_dataset

def noisy_pair_select(processed_dataset, st_idx, trainer, tokenizer, model, noise_threshold):
    """Find noisy pairs in the dataset."""
    logger.info(f"Searching for noisy pairs starting from index {st_idx}")
    for prompt_index in range(st_idx, len(processed_dataset)):
        one_sample = processed_dataset.select([prompt_index])

        if trainer.data_collator is not None:
            data_collator = trainer.data_collator
        else:
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            data_collator = DataCollatorForPreference(pad_token_id=pad_token_id)

        batch = data_collator([one_sample[i] for i in range(len(one_sample))])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        model.to(device)
        
        with torch.no_grad():
            outputs = trainer.concatenated_forward(trainer.model, batch)
        chosen_logps = outputs["chosen_logps"]  
        rejected_logps = outputs["rejected_logps"]
        
        if (chosen_logps > rejected_logps) and (chosen_logps - rejected_logps).abs() < noise_threshold and (chosen_logps - rejected_logps).abs() != 0:
            logger.info(f"Noisy pair found at index {prompt_index}: chosen_logps={chosen_logps.item():.4f}, rejected_logps={rejected_logps.item():.4f}")
            return batch, prompt_index + 1
        
        # Log progress periodically
        if prompt_index % 100 == 0:
            logger.info(f"Processed {prompt_index - st_idx} samples without finding a noisy pair")
    
    logger.info(f"Reached end of dataset (index {len(processed_dataset)}) without finding a noisy pair")
    return None, st_idx

class Comparison:
    def __init__(self, policy, padding_value, device='cuda'):
        self.policy = policy
        self.device = device
        self.policy = self.policy.to(device)
        self.policy = self.policy.to(torch.bfloat16)
        self.output_layer = policy.lm_head
        
        # Check if output layer is a WeightDeltaLayer or standard layer
        if hasattr(self.output_layer, 'base_weight'):
            # It's a WeightDeltaLayer
            self.original_params = self.output_layer.base_weight.flatten().clone()
        else:
            # Standard layer with weight attribute
            self.original_params = self.output_layer.weight.data.flatten().clone()
            
        self.padding_value = padding_value

    def pad_to_length(self, tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
        if tensor.size(dim) >= length:
            return tensor
        else:
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            return torch.cat(
                [
                    tensor,
                    pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                ],
                dim=dim,
            )

    def concatenated_inputs(self, batch: dict[str, Union[list, torch.LongTensor]], padding_value: int):
        output = {}
        output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        output["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )
        
        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        output["completion_input_ids"] = torch.cat(
            (
                self.pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                self.pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
            dim=0,
        )
        output["completion_attention_mask"] = torch.cat(
            (
                self.pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                self.pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
            dim=0,
        )

        return output
    
    def concatenated_forward(self, model: nn.Module, batch: dict[str, torch.LongTensor]):
        batch = self.concatenated_inputs(batch, padding_value=self.padding_value)
        prompt_input_ids = batch["prompt_input_ids"].to(self.device)
        completion_input_ids = batch["completion_input_ids"].to(self.device)
        prompt_attention_mask = batch["prompt_attention_mask"].to(self.device)
        completion_attention_mask = batch["completion_attention_mask"].to(self.device)

        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        
        # Create and shift mask ONCE
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
            dim=1,
        )
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()  # Single shift

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.to(torch.float32)

        # Generate labels and apply mask
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        labels[~loss_mask] = 0  # Use the already-shifted mask

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)  # Restore original positions
        
        all_logps = per_token_logps.sum(dim=-1)
        half = all_logps.shape[0] // 2
        
        return {
            "chosen_logps": all_logps[:half],
            "rejected_logps": all_logps[half:]
        }

    def update_params(self, params):
        with torch.no_grad():
            if hasattr(self.output_layer, 'set_delta'):
                return
                
            params_tensor = torch.tensor(params, device=self.device, dtype=torch.bfloat16)
            self.output_layer.weight.data = params_tensor.reshape(self.output_layer.weight.shape)
        
    def __call__(self, x: np.ndarray, y: np.ndarray, noisy_pairs, flag, flag2) -> tuple[int, int]:
        with torch.no_grad():
            if not hasattr(self.output_layer, 'set_delta'):
                self.update_params(x)
                
            x_chosen_logps = []
            x_rejected_logps = []
            for pairs in noisy_pairs:
                outputs = self.concatenated_forward(self.policy, pairs)
                x_chosen_logp, x_rejected_logp = outputs["chosen_logps"], outputs["rejected_logps"]
                x_chosen_logps.append(x_chosen_logp)
                x_rejected_logps.append(x_rejected_logp)

            if not hasattr(self.output_layer, 'set_delta'):
                self.update_params(y)
                
            y_chosen_logps = []
            y_rejected_logps = []
            for pairs in noisy_pairs:
                outputs = self.concatenated_forward(self.policy, pairs)
                y_chosen_logp, y_rejected_logp = outputs["chosen_logps"], outputs["rejected_logps"]
                y_chosen_logps.append(y_chosen_logp)
                y_rejected_logps.append(y_rejected_logp)

            if not hasattr(self.output_layer, 'set_delta'):
                self.output_layer.weight.data.copy_(self.original_params.reshape(self.output_layer.weight.shape))

            if flag2:
                for i in range(len(noisy_pairs)):
                    logger.info(f"Pair {i}:")
                    logger.info(f"  x_chosen_logps: {x_chosen_logps[i].item():.5f}, x_rejected_logps: {x_rejected_logps[i].item():.5f}")
                    logger.info(f"  y_chosen_logps: {y_chosen_logps[i].item():.5f}, y_rejected_logps: {y_rejected_logps[i].item():.5f}")
                return
            
            if flag:
                vote = 0
                comparison = 1
                for i in range(len(noisy_pairs)):
                    if y_chosen_logps[i].item() >  x_chosen_logps[i].item() and y_chosen_logps[i].item() - y_rejected_logps[i].item() > x_chosen_logps[i].item() - x_rejected_logps[i].item():    
                        vote += 1
                if vote > 0:
                    comparison = -1
                return comparison, 0

            vote = 0
            comparison = 1
            for i in range(len(noisy_pairs)):
                if y_chosen_logps[i].item() > x_chosen_logps[i].item() and y_rejected_logps[i].item() < x_rejected_logps[i].item():    
                    vote += 1
            if vote > 0:
                comparison = -1
            return comparison, 0

# Module-level function to fix pickling issue
def per_gpu_gradient_compute(local_rank, node_rank, policy, tokenizer, noisy_batch, m, grad_dir, world_size):
    """Per-GPU gradient computation function."""
    try:
        # Set device and process environment
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        
        # Initialize process group for this node
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        os.environ["RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://"
        )
        
        logger.info(f"Node {node_rank}, GPU {local_rank}: Initialized process group")
        
        # Move policy to this GPU
        local_policy = policy.to(device)
        padding_value = tokenizer.pad_token_id

        # Replace output layer with delta layer
        base_lm_head = local_policy.lm_head
        original_weights = base_lm_head.weight.data.clone()
        delta_layer = WeightDeltaLayer(base_lm_head)
        local_policy.lm_head = delta_layer
        
        # Convert to bfloat16 for efficiency
        local_policy = local_policy.to(torch.bfloat16)
        
        # Original weights for reference
        x = original_weights.flatten().to(torch.float32).cpu().numpy()
        local_u = np.zeros_like(x)

        # Create batches on this GPU
        gpu_noisy_batch = []
        for batch in noisy_batch:
            gpu_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    gpu_batch[k] = v.to(device)
                else:
                    gpu_batch[k] = v
            gpu_noisy_batch.append(gpu_batch)

        # Create comparison with delta-enabled model
        compare = Comparison(
            policy=local_policy,
            padding_value=padding_value,
            device=device
        )
        
        # Compute perturbations for this GPU
        count_negative_comparison = 0
        perturbation_scale = 0.0005  # Consistent value
        
        # Split perturbations across GPUs
        samples_per_gpu = m // 30
        samples_per_node = samples_per_gpu * 10
    
        
        # Process perturbations
        for i in range(samples_per_gpu):
            # Generate random direction
            z_np = np.random.normal(loc=0.0, scale=1.0, size=x.shape)
            z = torch.from_numpy(z_np).to(device, dtype=torch.bfloat16).reshape(delta_layer.delta.shape)
            
            # First pass with original weights
            delta_layer.reset()
            with torch.no_grad():
                x_outputs = []
                for pairs in gpu_noisy_batch:
                    outputs = compare.concatenated_forward(local_policy, pairs)
                    x_outputs.append((outputs["chosen_logps"], outputs["rejected_logps"]))
            
            # Second pass with perturbed weights
            delta_layer.set_delta(z, scale=perturbation_scale)
            with torch.no_grad():
                y_outputs = []
                for pairs in gpu_noisy_batch:
                    outputs = compare.concatenated_forward(local_policy, pairs)
                    y_outputs.append((outputs["chosen_logps"], outputs["rejected_logps"]))
            
            # Calculate comparison results
            y_i = 1  # Default comparison value
            for idx in range(len(gpu_noisy_batch)):
                x_chosen, x_rejected = x_outputs[idx]
                y_chosen, y_rejected = y_outputs[idx]
                
                if y_chosen.item() > x_chosen.item() and y_rejected.item() < x_rejected.item():
                    y_i = -1
                    break
            
            if y_i == -1:
                count_negative_comparison += 1
                
            # Update local gradient estimate for this GPU
            local_u += y_i * z_np
            # Progress reporting
            if i % 30 == 0:
                logger.info(f"Node {node_rank}, GPU {local_rank}: Processed {i} perturbations")
        
        # Convert numpy arrays to tensors for reduction
        u_tensor = torch.tensor(local_u, device=device)
        cnt_tensor = torch.tensor(count_negative_comparison, device=device)
        
        # Perform reduction to rank 0 of this node
        torch.distributed.reduce(u_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(cnt_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
        
        # Only rank 0 saves the results
        if local_rank == 0:
            # Get the aggregated results
            global_u = u_tensor.cpu().numpy()
            total_negative_comparisons = cnt_tensor.item()
            
            # Save the node-level aggregated gradient vector
            grad_info = {
                "node_rank": node_rank,
                "count_negative_comparison": total_negative_comparisons,
                "num_gpus": world_size,
                "samples_processed": samples_per_node
            }
            
            # Save the node-level aggregated gradient vector
            grad_file = os.path.join(grad_dir, f"node_grad_vector_{node_rank}.npy")
            np.save(grad_file, global_u)
            
            # Save metadata
            meta_file = os.path.join(grad_dir, f"node_grad_meta_{node_rank}.json")
            with open(meta_file, "w") as f:
                json.dump(grad_info, f)
                
            logger.info(f"Node {node_rank}: Saved aggregated gradient vector to {grad_file}")
            logger.info(f"Node {node_rank}: Processed total {samples_per_node} samples with {total_negative_comparisons} negative comparisons")
        
        # Clean up process group
        torch.distributed.destroy_process_group()
        
        return True
        
    except Exception as e:
        logger.error(f"Error in GPU {local_rank} on node {node_rank}: {e}")
        logger.error(traceback.format_exc())
        # Clean up process group
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return False

def compute_gradients_multi_gpu(policy, tokenizer, noisy_batch, m, node_rank, grad_dir):
    """Compute gradients for SCOBO update across all GPUs on a single node"""
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info(f"Node {node_rank}: Using {num_gpus} GPUs for gradient computation")
    
    # If only 1 GPU, run directly (no need for spawn)
    if num_gpus == 1:
        return per_gpu_gradient_compute(0, node_rank, policy, tokenizer, noisy_batch, m, grad_dir, num_gpus)
    
    # Launch multiple processes, one per GPU
    mp.set_start_method("spawn", force=True)
    mp.spawn(
        per_gpu_gradient_compute, 
        args=(node_rank, policy, tokenizer, noisy_batch, m, grad_dir, num_gpus),
        nprocs=num_gpus, 
        join=True
    )
    
    # Check if rank 0 produced the output file
    grad_file = os.path.join(grad_dir, f"node_grad_vector_{node_rank}.npy")
    meta_file = os.path.join(grad_dir, f"node_grad_meta_{node_rank}.json")
    
    if os.path.exists(grad_file) and os.path.exists(meta_file):
        return True
    else:
        logger.error(f"Node {node_rank}: Failed to produce gradient files")
        return False

def save_noisy_batch(noisy_batch, file_path):
    """Save noisy batch to disk for transfer to other nodes"""
    torch_data = {}
    for k, v in noisy_batch.items():
        if isinstance(v, torch.Tensor):
            torch_data[k] = v.cpu()
        else:
            torch_data[k] = v
    torch.save(torch_data, file_path)
    logger.info(f"Saved noisy batch to {file_path}")
    return True

def load_noisy_batch(file_path):
    """Load noisy batch from disk"""
    if not os.path.exists(file_path):
        logger.error(f"Noisy batch file {file_path} not found")
        return None
        
    try:
        torch_data = torch.load(file_path)
        logger.info(f"Loaded noisy batch from {file_path}")
        return torch_data
    except Exception as e:
        logger.error(f"Error loading noisy batch: {e}")
        logger.error(traceback.format_exc())
        return None

def find_noisy_pairs(model, tokenizer, dataset_offset, batch_size, noise_threshold, save_dir):
    """Find noisy pairs and save them for distribution to worker nodes"""
    logger.info("Loading dataset")
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    trainer, processed_dataset = trl_DPO_setup(model, tokenizer, dataset, tokenization=True)
    
    # Find noisy pairs
    st_idx = dataset_offset
    batches = []
    new_offset = st_idx
    
    for batch_idx in range(batch_size):
        noisy_batch, new_st_idx = noisy_pair_select(
            processed_dataset, st_idx, trainer, tokenizer, model, noise_threshold
        )
        
        if noisy_batch is None:
            logger.info("No more noisy pairs found. Resetting index.")
            st_idx = 0
            noisy_batch, new_st_idx = noisy_pair_select(
                processed_dataset, st_idx, trainer, tokenizer, model, noise_threshold
            )
            
        if noisy_batch is not None:
            batches.append(noisy_batch)
            st_idx = new_st_idx
            new_offset = new_st_idx
        else:
            logger.error("Could not find noisy batch after reset. Exiting.")
            return None, st_idx
    
    # Save all batches
    for i, batch in enumerate(batches):
        batch_file = os.path.join(save_dir, f"noisy_batch_{i}.pt")
        save_noisy_batch(batch, batch_file)
    
    # Save metadata about batches
    meta_file = os.path.join(save_dir, "batch_meta.json")
    with open(meta_file, "w") as f:
        json.dump({
            "num_batches": len(batches),
            "new_offset": new_offset
        }, f)
    
    return batches, new_offset

def aggregate_gradients(policy, tokenizer, noisy_batch, step_size, m, grad_dir):
    """Aggregate gradients from all nodes and apply the final update"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Aggregating gradients using device {device}")
    
    try:
        # Move policy to device
        policy = policy.to(device)
        padding_value = tokenizer.pad_token_id

        # Replace output layer with delta layer
        base_lm_head = policy.lm_head
        original_weights = base_lm_head.weight.data.clone()
        delta_layer = WeightDeltaLayer(base_lm_head)
        policy.lm_head = delta_layer
        
        # Convert to bfloat16 for efficiency
        policy = policy.to(torch.bfloat16)
        
        # Original weights for reference
        x = original_weights.flatten().to(torch.float32).cpu().numpy()
        
        # Load gradients from all nodes
        global_u = np.zeros_like(x)
        total_count_negative = 0
        total_nodes = 0
        
        for node_rank in range(3):  # 3 nodes
            grad_file = os.path.join(grad_dir, f"node_grad_vector_{node_rank}.npy")
            meta_file = os.path.join(grad_dir, f"node_grad_meta_{node_rank}.json")
            
            if os.path.exists(grad_file) and os.path.exists(meta_file):
                # Load node gradient vector
                node_u = np.load(grad_file)
                global_u += node_u
                
                # Load metadata
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    total_count_negative += meta["count_negative_comparison"]
                    
                logger.info(f"Loaded gradients from node {node_rank}: {meta['count_negative_comparison']} negative comparisons")
                total_nodes += 1
            else:
                logger.warning(f"Missing gradient files for node {node_rank}")
        
        if total_nodes == 0:
            logger.error("No gradient files found from any node. Cannot aggregate.")
            return None
            
        # Create batches on GPU once
        gpu_noisy_batch = []
        for batch in noisy_batch:
            gpu_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    gpu_batch[k] = v.to(device)
                else:
                    gpu_batch[k] = v
            gpu_noisy_batch.append(gpu_batch)
            
        # Create comparison for evaluation
        compare = Comparison(
            policy=policy,
            padding_value=padding_value,
            device=device
        )
        
        # Compute the full gradient
        norm_u = np.linalg.norm(global_u, 2)
        g_hat = global_u / norm_u if norm_u > 0 else np.zeros_like(global_u)
        logger.info(f"Number of entries in g_hat: {g_hat.size}")

        #Mistral 0.00022
        #Llama8BIns 0.00008 
        threshold = 0.00022 # Use consistent threshold
        negone_num_threshold = 330
        mask = np.abs(g_hat) >= threshold
        g_hat_sparse = g_hat * mask
        logger.info(f"Non-zero entries after thresholding: {np.count_nonzero(mask)}")

        # Test different step sizes using delta layer approach
        with torch.no_grad():
            flag = total_count_negative < negone_num_threshold
            if flag:
                logger.info(f"Using relaxed comparison (count_negative: {total_count_negative})")
            
            stepsizes = [0, 0.1, 0.5, 1, 2, 5, 10, 15]  # Consistent with original code
            for stepsize in stepsizes:
                logger.info(f"Testing stepsize: {stepsize}")
                
                # Reset delta layer
                delta_layer.reset()
                
                # Test this stepsize using delta approach
                test_delta = torch.from_numpy(-stepsize * g_hat_sparse).reshape(delta_layer.delta.shape).to(device, dtype=torch.bfloat16)
                delta_layer.set_delta(test_delta, scale=1.0)
                
                # Evaluate and print
                x_chosen_logps = []
                x_rejected_logps = []
                for pairs in gpu_noisy_batch:
                    outputs = compare.concatenated_forward(policy, pairs)
                    x_chosen_logps.append(outputs["chosen_logps"])
                    x_rejected_logps.append(outputs["rejected_logps"])
                
                for i in range(len(gpu_noisy_batch)):
                    logger.info(f"  chosen_logps: {x_chosen_logps[i].item():.5f}, rejected_logps: {x_rejected_logps[i].item():.5f}")
            
            # Final evaluation of step size 5 (consistent with original code)
            delta_layer.reset()
            
            # Evaluate baseline
            x_outputs = []
            for pairs in gpu_noisy_batch:
                outputs = compare.concatenated_forward(policy, pairs)
                x_outputs.append((outputs["chosen_logps"], outputs["rejected_logps"]))
            
            # Evaluate perturbed with step size 5
            delta_layer.set_delta(torch.from_numpy(-5 * g_hat_sparse).reshape(delta_layer.delta.shape).to(device, dtype=torch.bfloat16), scale=1.0)
            y_outputs = []
            for pairs in gpu_noisy_batch:
                outputs = compare.concatenated_forward(policy, pairs)
                y_outputs.append((outputs["chosen_logps"], outputs["rejected_logps"]))
            
            # Calculate comparison
            y_i = 1
            for idx in range(len(gpu_noisy_batch)):
                x_chosen, x_rejected = x_outputs[idx]
                y_chosen, y_rejected = y_outputs[idx]
                
                if y_chosen.item() > x_chosen.item() and y_rejected.item() < x_rejected.item():
                    y_i = -1
                    break
            
            # Apply final update
            updated_weights_flat = x - step_size * (total_count_negative/m) * g_hat_sparse
            logger.info(f"Update scale: {step_size * (total_count_negative/m)}")
            
            if flag or y_i == 1:
                updated_weights_flat = x
                logger.info("Didn't pass, keeping original weights")
                
            # Return the updated weights
            updated_weights = torch.tensor(updated_weights_flat, dtype=torch.bfloat16).reshape(original_weights.shape)
            return updated_weights
        
    except Exception as e:
        logger.error(f"Error aggregating gradients: {e}")
        logger.error(traceback.format_exc())
        return None
        
def save_lm_head_weights(model, file_path):
    """Save only the lm_head weights to a file"""
    if hasattr(model.lm_head, 'base_weight'):
        weights = model.lm_head.base_weight.cpu()
    else:
        weights = model.lm_head.weight.data.cpu()
    torch.save(weights, file_path)
    logger.info(f"Saved lm_head weights to {file_path}")
    return True

def load_and_apply_lm_head_weights(model, file_path):
    """Load lm_head weights from file and apply to model"""
    if not os.path.exists(file_path):
        logger.error(f"Weights file {file_path} not found")
        return False
        
    try:
        weights = torch.load(file_path)
        with torch.no_grad():
            if hasattr(model.lm_head, 'base_weight'):
                model.lm_head.base_weight.copy_(weights.to(model.lm_head.base_weight.device))
                model.lm_head.reset()
            else:
                model.lm_head.weight.data.copy_(weights.to(model.lm_head.weight.device))
        logger.info(f"Successfully applied weights from {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading weights: {e}")
        logger.error(traceback.format_exc())
        return False

def save_model_with_proper_lm_head(model, output_path, tokenizer):
    """Convert delta layer back to standard layer before saving"""
    # Check if using delta layer and convert back if needed
    if hasattr(model.lm_head, 'base_weight'):
        # Compute final weights from delta layer
        with torch.no_grad():
            final_weights = model.lm_head.base_weight + model.lm_head.scale * model.lm_head.delta
            
            # Store original layer
            original_lm_head = model.lm_head
            
            # Create standard linear layer with updated weights
            standard_lm_head = nn.Linear(final_weights.size(1), final_weights.size(0), bias=False)
            standard_lm_head.weight.data.copy_(final_weights)
            
            # Replace with standard layer
            model.lm_head = standard_lm_head
            
            # Save model
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            # Restore original layer (if needed for further processing)
            model.lm_head = original_lm_head
    else:
        # Already using standard layer
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    logger.info(f"Model saved to {output_path} in HuggingFace-compatible format")

def main():
    parser = argparse.ArgumentParser(description='Multi-node Multi-GPU Gradient Computation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output model')
    parser.add_argument('--grad_dir', type=str, required=True,
                        help='Directory to save/load gradients')
    parser.add_argument('--noisy_batch_dir', type=str, default=None,
                        help='Directory to save/load noisy batches')
    parser.add_argument('--weights_file', type=str, default=None,
                        help='Path to weights file for save/load operations')
    parser.add_argument('--run_all_iters', action='store_true',
                        help='Run all iterations, not just the ones in valid_list_llama_ins')
    parser.add_argument('--step_size', type=float, default=0.035,
                        help='Step size for optimization')
    parser.add_argument('--num_perturbation', type=int, default=1800,
                        help='Total number of perturbations across all nodes')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of noisy pairs to use')
    parser.add_argument('--noise_threshold', type=float, default=3.0,
                        help='Threshold for identifying noisy pairs')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Save model every N iterations')
    parser.add_argument('--node_rank', type=int, required=True,
                        help='Rank of this node (0=master, 1,2=workers)')
    parser.add_argument('--mode', type=str, choices=['find_pairs', 'compute', 'aggregate', 'apply_weights', 'full'], required=True,
                        help='Mode: find pairs, compute gradients, aggregate, apply weights, or full pipeline')
    parser.add_argument('--iteration', type=int, required=True,
                        help='Current iteration number')
    parser.add_argument('--dataset_offset', type=int, default=0,
                        help='Offset in the dataset to start from')
    parser.add_argument('--checkpoint_iters', type=str, default='36,40,45',
                   help='Comma-separated list of iterations to save checkpoints for')
    parser.add_argument('--temp_weights_file', type=str, default=None,
                    help='Path to save temp weights for next iteration')
    
    args = parser.parse_args()

    # Running first 10 noisy pairs for example
    valid_list_llama_ins = list(range(10))
    
    checkpoint_iters = []
    if args.checkpoint_iters:
        try:
            checkpoint_iters = [int(x.strip()) for x in args.checkpoint_iters.split(',')]
            logger.info(f"Will save checkpoints for iterations: {checkpoint_iters}")
        except ValueError:
            logger.warning(f"Invalid checkpoint iterations format: {args.checkpoint_iters}. Using default.")
            checkpoint_iters = [20]
    
    # Set up directories
    os.makedirs(args.grad_dir, exist_ok=True)
    iter_grad_dir = os.path.join(args.grad_dir, f"iter_{args.iteration}")
    os.makedirs(iter_grad_dir, exist_ok=True)
    
    if args.noisy_batch_dir is None:
        args.noisy_batch_dir = os.path.join(args.grad_dir, "noisy_batches")
    os.makedirs(args.noisy_batch_dir, exist_ok=True)
    
    # Set up iteration-specific noisy batch directory
    iter_batch_dir = os.path.join(args.noisy_batch_dir, f"iter_{args.iteration}")
    os.makedirs(iter_batch_dir, exist_ok=True)
    
    # Check if we should run this iteration
    if not args.run_all_iters and args.iteration not in valid_list_llama_ins and args.mode != 'find_pairs':
        logger.info(f"Skipping iteration {args.iteration} (not in valid list)")
        sys.exit(0)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # If weights file is specified, apply it
    if args.weights_file and os.path.exists(args.weights_file):
        logger.info(f"Loading weights from {args.weights_file}")
        load_and_apply_lm_head_weights(model, args.weights_file)

    # Ensure tokenizer has proper padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Mode 1: Find noisy pairs (master node only)
    if args.mode == 'find_pairs' or args.mode == 'full':
        if args.node_rank != 0:
            logger.error("Only node 0 (master) can find noisy pairs. Exiting.")
            sys.exit(1)
            
        logger.info("Finding noisy pairs for this iteration")
        noisy_batches, new_offset = find_noisy_pairs(
            model, tokenizer, args.dataset_offset, args.batch_size, 
            args.noise_threshold, iter_batch_dir
        )
        
        if noisy_batches is None:
            logger.error("Failed to find noisy pairs. Exiting.")
            sys.exit(1)
            
        # Save the new offset
        offset_file = os.path.join(args.grad_dir, "master_offset.txt")
        with open(offset_file, "w") as f:
            f.write(str(new_offset))
        
        logger.info(f"Found {len(noisy_batches)} noisy batch(es), saved to {iter_batch_dir}")
        logger.info(f"New dataset offset: {new_offset}")
    
    # Mode 2: Compute gradients
    elif args.mode == 'compute' or args.mode == 'full':
        # Load noisy batches for this iteration
        noisy_batches = []
        meta_file = os.path.join(iter_batch_dir, "batch_meta.json")
        
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
                num_batches = meta_data.get("num_batches", 0)
                
            for i in range(num_batches):
                batch_file = os.path.join(iter_batch_dir, f"noisy_batch_{i}.pt")
                batch = load_noisy_batch(batch_file)
                if batch is not None:
                    noisy_batches.append(batch)
        
        if len(noisy_batches) == 0:
            logger.error(f"No noisy batches found in {iter_batch_dir}. Exiting.")
            sys.exit(1)
        
        logger.info(f"Loaded {len(noisy_batches)} noisy batch(es) for gradient computation")
        
        # Compute gradients
        success = compute_gradients_multi_gpu(
            model, tokenizer, noisy_batches, args.num_perturbation, 
            args.node_rank, iter_grad_dir
        )
        
        if not success:
            logger.error("Gradient computation failed. Exiting.")
            sys.exit(1)
        
        logger.info(f"Node {args.node_rank}: Gradient computation completed successfully")
    
    # Mode 3: Aggregate gradients (master node only)
    elif args.mode == 'aggregate' or args.mode == 'full':
        if args.node_rank != 0:
            logger.error("Only node 0 (master) can aggregate gradients. Exiting.")
            sys.exit(1)
        
        # Load noisy batches for evaluation during aggregation
        noisy_batches = []
        meta_file = os.path.join(iter_batch_dir, "batch_meta.json")
        
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
                num_batches = meta_data.get("num_batches", 0)
                
            for i in range(num_batches):
                batch_file = os.path.join(iter_batch_dir, f"noisy_batch_{i}.pt")
                batch = load_noisy_batch(batch_file)
                if batch is not None:
                    noisy_batches.append(batch)
        
        if len(noisy_batches) == 0:
            logger.error(f"No noisy batches found in {iter_batch_dir}. Cannot evaluate updates.")
            sys.exit(1)
        
        # Wait until all node gradient files are present
        missing_files = True
        max_wait = 600  # 10 minutes
        wait_time = 0
        while missing_files and wait_time < max_wait:
            missing_files = False
            for node_rank in range(3):
                grad_file = os.path.join(iter_grad_dir, f"node_grad_vector_{node_rank}.npy")
                meta_file = os.path.join(iter_grad_dir, f"node_grad_meta_{node_rank}.json")
                if not os.path.exists(grad_file) or not os.path.exists(meta_file):
                    missing_files = True
                    logger.info(f"Waiting for gradients from node {node_rank}...")
                    break
            
            if missing_files:
                time.sleep(10)
                wait_time += 10
        
        if missing_files:
            logger.warning("Timed out waiting for gradient files. Proceeding with available gradients.")
        
        # Aggregate gradients and update model
        updated_weights = aggregate_gradients(
            model, tokenizer, noisy_batches, args.step_size, 
            args.num_perturbation, iter_grad_dir
        )
        
        if updated_weights is not None:
            # Apply the updated weights based on the layer type
            with torch.no_grad():
                if hasattr(model.lm_head, 'base_weight'):
                    # It's a WeightDeltaLayer, update the base_weight
                    model.lm_head.base_weight.copy_(updated_weights)
                    # Reset any delta
                    model.lm_head.reset()
                else:
                    # Standard layer
                    model.lm_head.weight.data = updated_weights
            logger.info("Model weights updated successfully")
            
            # Save weights file if specified for worker nodes to use
            if args.weights_file:
                save_lm_head_weights(model, args.weights_file)
                logger.info(f"Saved weights to {args.weights_file}")
            
            # Save temp weights for next iteration if specified
            if args.temp_weights_file:
                save_lm_head_weights(model, args.temp_weights_file)
                logger.info(f"Saved temp weights to {args.temp_weights_file}")
            
            # Save checkpoint only if needed (full model in HuggingFace format)
            if args.iteration in checkpoint_iters:
                checkpoint_dir = f"{args.output_path}_{args.iteration}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_model_with_proper_lm_head(model, checkpoint_dir, tokenizer)
                logger.info(f"Checkpoint saved at {checkpoint_dir}")
                
                # Also save to output_path for loading in next iteration
                os.makedirs(args.output_path, exist_ok=True)
                save_model_with_proper_lm_head(model, args.output_path, tokenizer)
                logger.info(f"Model also saved to {args.output_path} for next iteration")
            else:
                logger.info(f"Skipping full model checkpoint for iteration {args.iteration}")

        else:
            logger.error("Failed to aggregate gradients. Model not updated.")
            sys.exit(1)
        
        logger.info(f"Iteration {args.iteration} completed successfully")
    
    # Mode 4: Apply weights (worker nodes)
    elif args.mode == 'apply_weights':
        if not args.weights_file:
            logger.error("Weights file must be provided for apply_weights mode")
            sys.exit(1)
            
        success = load_and_apply_lm_head_weights(model, args.weights_file)
        if not success:
            logger.error("Failed to apply weights. Exiting.")
            sys.exit(1)
        
        # Save model with updated weights
        os.makedirs(args.output_path, exist_ok=True)
        model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        logger.info(f"Model saved to {args.output_path} with updated weights")
        logger.info(f"Node {args.node_rank}: Successfully applied weights")

if __name__ == "__main__":
    main()
