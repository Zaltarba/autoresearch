"""
Time series autoresearch training script.
This is the file the agent modifies each experiment.

Usage: python ts_model.py
"""

import time
import math
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

from ts_prepare import (
    TIME_BUDGET, SEQ_LEN, LABEL_LEN, PRED_LEN,
    FREQ, FEATURES, TARGET,
    make_dataloaders, evaluate_mse,
)
from models import model_dict

# ---------------------------------------------------------------------------
# Experiment configuration — agent modifies this section
# ---------------------------------------------------------------------------

# Select model: iTransformer | Transformer | iInformer | Informer |
#               iReformer | Reformer | iFlowformer | Flowformer |
#               iFlashformer | Flashformer
MODEL_NAME = "iTransformer"

@dataclass
class Config:
    # Data (must match ts_prepare.py constants)
    seq_len:    int   = SEQ_LEN
    label_len:  int   = LABEL_LEN
    pred_len:   int   = PRED_LEN
    freq:       str   = FREQ
    features:   str   = FEATURES
    target:     str   = TARGET

    # I/O dimensions — update if you change the dataset in ts_prepare.py
    enc_in:  int = 7   # number of input variates (ETTh1 has 7)
    dec_in:  int = 7
    c_out:   int = 7

    # Model architecture
    d_model:    int   = 128
    n_heads:    int   = 8
    e_layers:   int   = 2
    d_layers:   int   = 1
    d_ff:       int   = 128
    dropout:    float = 0.1
    factor:     int   = 1
    distil:     bool  = True
    activation: str   = "gelu"

    # Embedding
    embed: str = "timeF"

    # iTransformer-specific
    use_norm:       bool = True
    class_strategy: str  = "projection"

    # Transformer/enc-dec specific
    channel_independence: bool = False

    # Output
    output_attention: bool = False

    # Training
    batch_size:    int   = 32
    learning_rate: float = 1e-4

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = model_dict[MODEL_NAME]
    model = model_cls(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_loader, val_loader = make_dataloaders(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    model.train()
    step = 0
    epoch = 0
    train_start = time.time()

    print(f"Model: {MODEL_NAME}  |  params: {num_params/1e6:.2f}M  |  device: {device}")
    print(f"Training for {TIME_BUDGET}s...")

    while time.time() - train_start < TIME_BUDGET:
        epoch += 1
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            if time.time() - train_start >= TIME_BUDGET:
                break

            batch_x      = batch_x.float().to(device)
            batch_y      = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -PRED_LEN:, :]).float()
            dec_inp = torch.cat([batch_y[:, :LABEL_LEN, :], dec_inp], dim=1).float().to(device)

            optimizer.zero_grad()
            output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if isinstance(output, tuple):
                output = output[0]

            loss = criterion(output[:, -PRED_LEN:, :], batch_y[:, -PRED_LEN:, :])
            loss.backward()
            optimizer.step()
            step += 1

    training_seconds = time.time() - train_start

    eval_start = time.time()
    val_mse = evaluate_mse(model, val_loader, device)
    total_seconds = time.time() - train_start + (time.time() - eval_start)

    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / 1024 / 1024
                    if torch.cuda.is_available() else 0.0)

    print("---")
    print(f"val_mse:          {val_mse:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {time.time() - train_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params/1e6:.2f}")
    print(f"model:            {MODEL_NAME}")
    print(f"d_model:          {config.d_model}")
    print(f"e_layers:         {config.e_layers}")
    print(f"batch_size:       {config.batch_size}")
    print(f"learning_rate:    {config.learning_rate}")


if __name__ == "__main__":
    train()
