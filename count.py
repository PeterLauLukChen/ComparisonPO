import re
import numpy as np
import sys
import matplotlib.pyplot as plt

def parse_log_file(file_path):
    """
    Parse the log file to extract negative comparison counts for each iteration and run.
    Returns a 2D array with the summed negative comparisons.
    """
    # Initialize a 10x11 array for storing results (10 iterations, 11 runs)
    results = np.zeros((10, 11), dtype=int)
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find all run blocks
    run_blocks = re.split(r'={20,}\nProcessing iteration 0\n={20,}', content)
    
    # First block is initialization, skip it
    if len(run_blocks) > 1:
        run_blocks = run_blocks[1:]
    
    # Process each run
    for run_idx, run_block in enumerate(run_blocks):
        if run_idx >= 11:  # Only process up to 11 runs
            break
            
        # Find all iteration blocks within this run
        iteration_blocks = re.split(r'={20,}\nProcessing iteration (\d+)\n={20,}', run_block)
        
        # First iteration (0) is already in progress when the block starts
        iterations = [iteration_blocks[0]]
        
        # Add remaining iterations
        for i in range(1, len(iteration_blocks), 2):
            if i+1 < len(iteration_blocks):
                iter_num = int(iteration_blocks[i])
                iterations.append(iteration_blocks[i+1])
        
        # Process each iteration
        for iter_idx, iter_block in enumerate(iterations):
            if iter_idx >= 10:  # Only process up to 10 iterations
                break
                
            # Find negative comparison lines
            neg_comp_lines = re.findall(r'Loaded gradients from node \d+: (\d+) negative comparisons', iter_block)
            
            # Sum the negative comparisons
            total_neg_comp = sum(int(count) for count in neg_comp_lines)
            
            # Store the result
            if run_idx < 11 and iter_idx < 10:
                results[iter_idx, run_idx] = total_neg_comp
    
    return results

def plot_results(results):
    """
    Create a heatmap visualization of the results.
    """
    plt.figure(figsize=(14, 10))
    plt.imshow(results, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Sum of Negative Comparisons')
    plt.title('Negative Comparisons by Iteration and Run')
    plt.xlabel('Run Number')
    plt.ylabel('Iteration Number')
    
    # Add text annotations
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            if results[i, j] > 0:  # Only show non-zero values
                plt.text(j, i, str(results[i, j]), 
                         ha="center", va="center", 
                         color="white" if results[i, j] > results.max()/2 else "black")
    
    # Set tick labels
    plt.xticks(range(11), [f'Run {i+1}' for i in range(11)])
    plt.yticks(range(10), [f'Iter {i}' for i in range(10)])
    
    plt.tight_layout()
    plt.savefig('negative_comparisons_heatmap.png')
    print("Heatmap saved as 'negative_comparisons_heatmap.png'")

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_negative_comparisons.py <log_file_path>")
        return
    
    file_path = sys.argv[1]
    results = parse_log_file(file_path)
    
    # Print the results as a 10x11 array
    print("Sum of Negative Comparisons (10 iterations x 11 runs):")
    for i in range(10):
        row = ' '.join(f"{results[i, j]:4d}" for j in range(11))
        print(f"Iteration {i}: [{row} ]")
    
    # Calculate statistics
    print("\nStatistics:")
    print(f"Average per iteration: {results.mean(axis=1)}")
    print(f"Total per run: {results.sum(axis=0)}")
    print(f"Grand total: {results.sum()}")
    
    # Plot the results
    try:
        plot_results(results)
    except Exception as e:
        print(f"Error creating plot: {e}")
        print("Continuing without plot generation.")

if __name__ == "__main__":
    main()