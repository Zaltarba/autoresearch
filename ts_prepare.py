"""
Fixed evaluation harness for time series autoresearch experiments.
DO NOT MODIFY — this file defines the ground-truth metric and dataset configuration.

Provides:
  - Dataset and forecasting constants (fixed)
  - make_dataloaders(config) -> (train_loader, val_loader)
  - evaluate_mse(model, val_loader, device) -> float  (lower is better)
"""

import time
import numpy as np
import torch
import torch.nn as nn

from data_provider.data_factory import data_provider

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # wall-clock training seconds per experiment (5 min)

# Dataset
DATA_ROOT    = "datasets/iTransformer_datasets/ETT-small"
DATA_FILE    = "ETTh1.csv"
DATA_TYPE    = "ETTh1"     # key into data_provider data_dict
FREQ         = "h"         # hourly
FEATURES     = "M"         # multivariate -> multivariate
TARGET       = "OT"        # target column (used for MS mode)
NUM_WORKERS  = 0

# Forecasting setup
SEQ_LEN      = 96          # look-back window
LABEL_LEN    = 48          # decoder start token length (for enc-dec models)
PRED_LEN     = 96          # forecast horizon


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class _Args:
    """Minimal args object consumed by data_provider."""
    def __init__(self, config, flag):
        self.root_path  = DATA_ROOT
        self.data_path  = DATA_FILE
        self.data       = DATA_TYPE
        self.freq       = FREQ
        self.features   = FEATURES
        self.target     = TARGET
        self.seq_len    = SEQ_LEN
        self.label_len  = LABEL_LEN
        self.pred_len   = PRED_LEN
        self.embed      = config.embed
        self.batch_size = config.batch_size
        self.num_workers = NUM_WORKERS


def make_dataloaders(config):
    """Returns (train_loader, val_loader)."""
    train_args = _Args(config, 'train')
    val_args   = _Args(config, 'val')
    _, train_loader = data_provider(train_args, 'train')
    _, val_loader   = data_provider(val_args,   'val')
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Evaluation — ground truth metric, do not modify
# ---------------------------------------------------------------------------

def evaluate_mse(model, val_loader, device):
    """Evaluate model on the fixed validation split. Returns val_mse (lower is better)."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
            batch_x       = batch_x.float().to(device)
            batch_y       = batch_y.float().to(device)
            batch_x_mark  = batch_x_mark.float().to(device)
            batch_y_mark  = batch_y_mark.float().to(device)

            # decoder input: zero-padded for enc-only models, label prefix for enc-dec
            dec_inp = torch.zeros_like(batch_y[:, -PRED_LEN:, :]).float()
            dec_inp = torch.cat([batch_y[:, :LABEL_LEN, :], dec_inp], dim=1).float().to(device)

            output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # handle (output, attns) tuple
            if isinstance(output, tuple):
                output = output[0]

            pred = output[:, -PRED_LEN:, :].detach().cpu().numpy()
            true = batch_y[:, -PRED_LEN:, :].detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    val_mse = float(np.mean((preds - trues) ** 2))
    model.train()
    return val_mse
