# [NeurIPS 2025] ComPO: Preference Alignment via Comparison Oracles 

A novel, lightweight preference optimization algorithm for LLM post training. See the paper at [https://arxiv.org/abs/2505.05465](https://arxiv.org/abs/2505.05465). 

## Example implementation for multi-nodes training (30 GPUs parallel perturbation)

Welcome to **ComPO**—a lightweight framework for distributed multi‑node training. This repository contains the shell scripts and Python modules required to launch, coordinate, and monitor large‑scale ComPO training runs across a master node and multiple worker nodes.

---

If you find the idea interesting, please cite

```bibtex
@inproceedings{Chen-2025-compo,
  title={ComPO: Preference Alignment via Comparison Oracles},
  author={Peter Chen and Xi Chen and Wotao Yin and Tianyi Lin},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=0lNwIIHWhZ}
}
```

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Repository Setup](#repository-setup)
4. [Configuring Your Run](#configuring-your-run)
5. [Launching Training](#launching-training)
6. [Hyper‑parameter Reference](#hyper-parameter-reference)
7. [Example Log](#example-log)
8. [Troubleshooting](#troubleshooting)
9. [License](#license)

---

## Overview
ComPO extends existing language‑model fine‑tuning pipelines with a **comparison‑based objective** from comparison oracles that requires two key constants—`gradient entry threshold` (filtering out noisy entry to ensure sparse gradien update) and `sample gradient threshold` (number of successful comparison oracle returned)—to adapt to the model size and perturbation count during zeroth-order gradient estimation for preference training. The framework assumes:

- **Symmetric file‑system layout** on every node (same absolute path).
- Password‑less SSH (or properly configured key‑based auth) from **master ➜ workers**.
- GCC ≥ 7.5 and Python ≥ 3.10.
- All nodes can scp with each other (pubkey between nodes is added to each). We don't need ray for parallelization in ComPO.

---

## Repository Setup
1. **Clone** (or `rsync`) this repository to **every** node—master **and** workers—**using exactly the same path**  

2. Install dependencies 

```bash
pip install -r requirements.txt
```

3. Download the base model used for ComPO augmentation on all nodes

---

## Configuring Your Run

The main training configuration is located in `run_training.sh` on the master node. The worker configuration is located in `run_worker.sh` on the master node.
For `run_worker.sh`,you should modify the configuration **only** on the master node. This script also serves as the working script for the worker nodes—once training is launched from the master, the script will automatically `scp` the updated version to all worker nodes.

Next, refer to `compute_gradients.py`, which contains the core training logic for ComPO.

There are two specific constants that must be adjusted depending on the model being trained. The default values are tuned for Mistral:

1. `Threshold`, which is used on line 603, with a default value of `0.00022`.  
2. `Negone_num_threshold`, which is used on line 604, with a default value of `330`, assuming the number of perturbations is `1400`.

When adapting to other models, both parameters should be adjusted accordingly.

We provide the following reference values for LLaMA and Gemma (with 1600 perturbations):

- `Negone_num_threshold = 370`
- `Threshold = 0.00008`

---

## Launching Training

After editing these files on the master node, run the following on the master node:

```bash
bash run_training.sh
```

--- 

## Example Log

An example session is provided in Example_log.out. 
