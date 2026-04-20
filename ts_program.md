# ts_autoresearch

Autonomous time series forecasting experiments. The agent modifies `ts_model.py` to improve **val_mse** (lower is better) within a fixed 5-minute training budget per experiment.

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date and the model (e.g. `apr20-itransformer`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files** for full context:
   - `ts_prepare.py` — fixed constants, data loading, evaluation harness. **Do not modify.**
   - `ts_model.py` — the file you modify. Model selection, Config dataclass, training loop.
4. **Initialize results.tsv**: create it with just the header row.
5. **Confirm and go.**

## Target model

At the start of each session, the user tells you which model to improve (e.g. "start from iTransformer"). Set `MODEL_NAME` in `ts_model.py` to that model and run the baseline first.

Available models (all in `models/`):
- `iTransformer` — inverted attention, encoder-only (best default)
- `Transformer` — vanilla enc-dec
- `iInformer` — inverted + ProbSparse attention
- `Informer` — enc-dec + ProbSparse + distilling
- `iReformer` — inverted + LSH attention
- `Reformer` — encoder-only + LSH attention
- `iFlowformer` — inverted + Flow attention
- `Flowformer` — enc-dec + Flow attention in encoder
- `iFlashformer` — inverted + block Flash attention
- `Flashformer` — enc-dec + block Flash attention

## Experimentation

**What you CAN do:**
- Modify `ts_model.py` — everything is fair game: `MODEL_NAME`, `Config` dataclass values (d_model, n_heads, e_layers, d_ff, dropout, batch_size, learning_rate, etc.), the training loop, optimizer, loss function, LR schedule.
- Modify individual model files in `models/` to try architectural changes (add residuals, change normalization, etc.)

**What you CANNOT do:**
- Modify `ts_prepare.py`. It is read-only. It defines the fixed evaluation, data splits, TIME_BUDGET, SEQ_LEN, PRED_LEN.
- Change the dataset or evaluation split. `evaluate_mse()` in `ts_prepare.py` is the ground truth.
- Install new packages.

**The goal: lowest val_mse on ETTh1, pred_len=96.**

**Simplicity criterion**: a small improvement that adds ugly complexity is not worth it. Removing something and getting equal/better results is always a win.

## Running experiments

```bash
bash script.sh
```

This runs `python ts_model.py > run.log 2>&1` and greps key metrics. To diagnose a crash:
```bash
tail -n 50 run.log
```

## Output format

After a successful run, the script prints:
```
val_mse:          0.043210
training_seconds: 300.1
total_seconds:    312.4
peak_vram_mb:     4200.0
num_steps:        1500
num_params_M:     2.30
model:            iTransformer
d_model:          128
e_layers:         2
batch_size:       32
learning_rate:    0.0001
```

## Logging results

Log to `results.tsv` (tab-separated, NOT comma-separated). Keep it **untracked** — do not commit it.

```
commit	val_mse	memory_gb	status	description
a1b2c3d	0.043210	4.1	keep	baseline iTransformer d_model=128
b2c3d4e	0.041500	4.1	keep	increase d_model to 256
c3d4e5f	0.044000	4.1	discard	switch to Transformer enc-dec
```

Columns:
1. git commit hash (7 chars)
2. val_mse (6 decimal places) — use 0.000000 for crashes
3. peak_vram_mb / 1024, rounded to .1f — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of the change

## The experiment loop

LOOP FOREVER:

1. Check git state (current branch/commit)
2. Modify `ts_model.py` (or a model file in `models/`) with one experimental idea
3. `git commit`
4. `bash script.sh` (redirects everything to run.log automatically)
5. Read metrics: `grep "^val_mse:\|^peak_vram_mb:" run.log`
6. If grep is empty → crash. Run `tail -n 50 run.log`, attempt fix. After 2-3 failed attempts, give up and move on.
7. Record in `results.tsv`
8. If val_mse improved (lower) → keep the commit, advance the branch
9. If val_mse equal or worse → `git reset --hard HEAD~1` to discard

**Timeout**: if a run exceeds 10 minutes, kill it (Ctrl+C) and treat as failure.

**NEVER STOP**: once the loop has started, do not pause to ask if you should continue. Run until manually interrupted. If you run out of ideas, think harder — try different models, different hyperparameter scales, LR schedules, architectural tweaks inside model files.
