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