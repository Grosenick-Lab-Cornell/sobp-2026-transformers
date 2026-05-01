---
description: "SLURM cluster operations. Usage: /slurm <subcommand> [args]"
---

SLURM cluster operations dispatcher.

Parse `$ARGUMENTS` to determine the subcommand and remaining args. The first word is the subcommand.

<!--
  Placeholders to fill in before use:
    <SSH_HOST>     SSH alias for the cluster login node (defined in ~/.ssh/config)
    <PROJECT_DIR>  Absolute path to your project on the cluster (e.g., /projects/abc123/MyRepo)
    <SCRATCH_DIR>  Absolute path to scratch storage on the cluster (e.g., /scratch/abc123)
-->

## Subcommands

### `status [--running] [--pattern=NAME] [--json]`

Check SLURM job status and training progress.

```bash
ssh <SSH_HOST> "\$HOME/bin/mrl-status {remaining_args}"
```

The script is pure bash --- no venv needed. It outputs a compact human-readable format.
With `--json`, it outputs JSON only for programmatic use.

### `submit <config_path> [--partition=X] [--time=HH:MM:SS] [--time-min=HH:MM:SS] [--gpus=N] [--cpus=N] [--mem=XG] [--exclude=node1,node2]`

Submit experiment to the cluster. Atomic operation: syncs code, snapshots it, submits.

1. Sync code to the cluster:
```bash
rsync -avz --delete \
  --exclude='/data' --exclude='outputs' --exclude='.venv*' \
  --exclude='__pycache__' --exclude='*.mmap' --exclude='*.pt' \
  --exclude='*.pth' --exclude='*.bin' --exclude='*.npy' \
  --exclude='.git' --exclude='*.egg-info' --exclude='build' \
  --exclude='dist' --exclude='.obsidian' --exclude='memory' \
  --exclude='.claude' --exclude='.env' --exclude='docs' \
  --exclude='tests' --exclude='tmp' \
  ./ <SSH_HOST>:<PROJECT_DIR>/
```

2. Run mrl-submit:
```bash
ssh <SSH_HOST> "source <PROJECT_DIR>/.venv/bin/activate && \$HOME/bin/mrl-submit {config_path} {remaining_flags}"
```
`{config_path}` is the first arg after `submit` (relative to project root, e.g., `configs/pretrain_exp_a.yaml`). If missing, ask which config to use.

### `eval <script_path> [script_args...] [--partition=X] [--time=HH:MM:SS]`

Submit an evaluation/probe job to the cluster. Runs any Python script as a SLURM job.

**Naming convention**: Job names follow `eval_{exp_descriptor}` where `exp_descriptor` drops any `exp_YYYYMMDD_` prefix. Auto-derive from the experiment dir/checkpoint path in the args.

**Parsing**: `{script_path}` is the first arg after `eval`. Remaining args are split into:
- SLURM flags: `--partition`, `--time` (consumed by the skill, mapped to `--slurm-partition`, `--slurm-time`)
- Everything else: passed through to the script

If no `--time` given, default to `02:00:00`.

Auto-derive the job name: scan the script args for a path containing `exp_` (e.g., `outputs/exp_20260311_my_experiment` or a checkpoint path). Extract the experiment name, strip `exp_YYYYMMDD_` prefix, prepend `eval_`. If no experiment found in args, use `eval_{script_basename}`.

1. Sync code to the cluster (same rsync as `push`):
```bash
rsync -avz --delete \
  --exclude='/data' --exclude='outputs' --exclude='.venv*' \
  --exclude='__pycache__' --exclude='*.mmap' --exclude='*.pt' \
  --exclude='*.pth' --exclude='*.bin' --exclude='*.npy' \
  --exclude='.git' --exclude='*.egg-info' --exclude='build' \
  --exclude='dist' --exclude='.obsidian' --exclude='memory' \
  --exclude='.claude' --exclude='.env' --exclude='docs' \
  --exclude='tests' --exclude='tmp' \
  ./ <SSH_HOST>:<PROJECT_DIR>/
```

2. Build the mrl-eval command, mapping skill flags to `--slurm-*` flags:
```bash
ssh <SSH_HOST> "source <PROJECT_DIR>/.venv/bin/activate && \$HOME/bin/mrl-eval {script_path} {script_args} --slurm-partition={partition} --slurm-time={time} --slurm-job-name={job_name}"
```

### `pull [experiment_name]`

Pull experiment outputs from the cluster.

```bash
rsync -avz --progress \
  --exclude='*.mmap' \
  --exclude='*.bin' \
  --exclude='code' \
  <SSH_HOST>:<SCRATCH_DIR>/outputs/{experiment_or_all}/ ./outputs/{experiment_or_all}/
```
No `--delete` --- preserve local outputs. Excludes the `code/` snapshot (large, not needed locally). Includes `*.pt` and `*.npy` (checkpoints and eval artifacts).

If no experiment name given, pull ALL outputs.

### `push`

Sync local code to the cluster.

```bash
rsync -avz --delete \
  --exclude='/data' --exclude='outputs' --exclude='.venv*' \
  --exclude='__pycache__' --exclude='*.mmap' --exclude='*.pt' \
  --exclude='*.pth' --exclude='*.bin' --exclude='*.npy' \
  --exclude='.git' --exclude='*.egg-info' --exclude='build' \
  --exclude='dist' --exclude='.obsidian' --exclude='memory' \
  --exclude='.claude' --exclude='.env' \
  --exclude='docs' --exclude='tests' --exclude='tmp' \
  ./ <SSH_HOST>:<PROJECT_DIR>/
```

Report what was synced (file count and total size from rsync output).

### `nodes`

Show cluster node availability.

```bash
ssh <SSH_HOST> "\$HOME/bin/mrl-nodes"
```

### `run <command...>`

Run an arbitrary command on the cluster.

Route based on the first word of `{command}`:
- If it starts with `python`, `pytest`, or `uv`: use the **Python path**
- Otherwise: use the **SLURM path**

**SLURM path** (handles module loading):
```bash
ssh <SSH_HOST> "\$HOME/bin/mrl-run {command}"
```

**Python path** (activates venv, sets PYTHONPATH):
```bash
ssh <SSH_HOST> "source <PROJECT_DIR>/.venv/bin/activate && cd <PROJECT_DIR> && PYTHONPATH=. {command}"
```

If no command provided, ask the user what to run.

### `setup`

One-time environment setup. Run once before first job submission.

1. Create directory structure:
```bash
ssh <SSH_HOST> 'mkdir -p <PROJECT_DIR> <SCRATCH_DIR>/outputs <SCRATCH_DIR>/data ~/bin ~/.mrl'
```

2. Create symlinks:
```bash
ssh <SSH_HOST> 'ln -sfn <SCRATCH_DIR>/data <PROJECT_DIR>/data'
ssh <SSH_HOST> 'ln -sfn <SCRATCH_DIR>/outputs <PROJECT_DIR>/outputs'
```

3. Deploy mrl-* scripts and config:
```bash
ssh <SSH_HOST> 'ln -sf <PROJECT_DIR>/scripts/mrl-run $HOME/bin/mrl-run'
ssh <SSH_HOST> 'ln -sf <PROJECT_DIR>/scripts/mrl-submit $HOME/bin/mrl-submit'
ssh <SSH_HOST> 'ln -sf <PROJECT_DIR>/scripts/mrl-status $HOME/bin/mrl-status'
ssh <SSH_HOST> 'ln -sf <PROJECT_DIR>/scripts/mrl-nodes $HOME/bin/mrl-nodes'
ssh <SSH_HOST> 'ln -sf <PROJECT_DIR>/scripts/mrl-eval $HOME/bin/mrl-eval'
scp scripts/config.yaml <SSH_HOST>:~/.mrl/config.yaml
```

4. Verify scripts work:
```bash
ssh <SSH_HOST> '$HOME/bin/mrl-run squeue --version'
```

5. Sync code --- execute the `push` subcommand above.

6. Check if uv is installed:
```bash
ssh <SSH_HOST> 'which uv 2>/dev/null || echo "NOT_FOUND"'
```
If NOT_FOUND:
```bash
ssh <SSH_HOST> 'curl -LsSf https://astral.sh/uv/install.sh | sh'
```

7. Create venv and install dependencies (NOT the package --- PYTHONPATH handles that via code snapshots):
```bash
ssh <SSH_HOST> 'cd <PROJECT_DIR> && uv venv && uv sync --no-install-project'
```

8. Verify installation:
```bash
ssh <SSH_HOST> '<PROJECT_DIR>/.venv/bin/python -c "import torch; print(f\"torch={torch.__version__}, cuda={torch.cuda.is_available()}\")"'
```

9. Verify preprocessed data (adjust filenames to your project):
```bash
ssh <SSH_HOST> 'ls <SCRATCH_DIR>/data/'
```

10. Report setup status.

## Output rules

- Put all command output in a code block, exactly as returned. Do NOT editorialize, reformat into tables, summarize, or strip away actual output.
- rsync file lists are noise --- only report totals (files transferred, size).
- If a command fails, report the error verbatim and stop.
