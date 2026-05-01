# slurm-whisperer

A toolkit for managing SLURM clusters: `/slurm push`, `submit`,
`status`, `pull`, `nodes`, `run`, `setup`. Each subcommand wraps an `ssh` or
`rsync` call to your cluster login node. With `ControlMaster` enabled in
`~/.ssh/config` (see Prerequisites), every call piggybacks on a single
authenticated SSH connection --- no re-authenticating per command. A thin
executor subagent runs the commands and returns raw output, keeping
cluster-op noise out of your main Claude context.

**What this is:** a slash command (`.claude/commands/slurm.md`) plus a thin
executor subagent (`.claude/agents/slurm-ops.md`). Install both.

## What's in here

```
.claude/
  agents/slurm-ops.md     # thin relay subagent (runs commands, returns output verbatim)
  commands/slurm.md       # /slurm dispatcher: push, pull, submit, eval, status, nodes, run, setup
scripts/
  mrl-run, mrl-submit, mrl-status, mrl-nodes, mrl-eval   # config-driven helpers (run on the cluster)
  config.yaml.example     # cluster config template
```

The split-design rationale:

- `commands/slurm.md` is the *recipe book* --- the canonical specification of every
  rsync/ssh command in the workflow.
- `agents/slurm-ops.md` is the *executor* --- a thin subagent (with
  `bypassPermissions`) that reads the recipe and runs the commands verbatim,
  returning raw output with no commentary. This keeps cluster-op noise (rsync
  file lists, ssh banners) out of your main orchestrator context.

## Prerequisites

1. **SSH alias** to your cluster login node in `~/.ssh/config`, e.g.:
   ```
   Host mycluster
     HostName login.mycluster.example.edu
     User youruser
     ControlMaster auto
     ControlPath ~/.ssh/cm_%h_%p_%r
     ControlPersist 10m
   ```
2. **`uv`** installed locally (https://astral.sh/uv) for managing the Python venv.
3. A SLURM project/account that can submit jobs.

## Install

1. **Copy** the `.claude/` and `scripts/` directories into your project root.
   (They merge with any existing `.claude/` you have.)

2. **Fill in placeholders.** Open `.claude/commands/slurm.md` and replace:
   - `<SSH_HOST>` --- your SSH alias from step 1 above
   - `<PROJECT_DIR>` --- absolute path to your project on the cluster (e.g., `/projects/abc123/MyRepo`)
   - `<SCRATCH_DIR>` --- absolute scratch path (e.g., `/scratch/abc123`)

   Then `cp scripts/config.yaml.example scripts/config.yaml` and fill in
   `user`, `project_dir`, `scratch_dir`, and `account` for each partition.

3. **Run setup once** from your local Claude Code session:
   ```
   /slurm setup
   ```
   This creates the cluster directory structure, deploys `mrl-*` to `$HOME/bin`,
   uploads `~/.mrl/config.yaml`, syncs your code, and creates the venv.

## Daily use

```
/slurm push                           # sync local code to cluster
/slurm submit configs/my_exp.yaml     # snapshot + submit
/slurm status                         # squeue + training progress for your jobs
/slurm pull exp_20260101_my_run       # rsync outputs back
/slurm nodes                          # cluster availability
/slurm run nvidia-smi                 # ad-hoc command
/slurm eval scripts/my_probe.py outputs/exp_X --time=01:00:00
```

## Customizing

The `submit`, `eval`, and `push` subcommands all share the same rsync exclude
list. If your project has different "do not sync" patterns (e.g., a different
data dir name), edit them in all three places in `.claude/commands/slurm.md`.

If you place the `mrl-*` scripts somewhere other than `scripts/` in your
project, update the symlink source paths in the `setup` subcommand of
`.claude/commands/slurm.md` to match.

## Why "slurm-whisperer"?

Because submitting to a SLURM cluster shouldn't require remembering 14 sbatch
flags every time. Tell the whisperer what you want; it talks to the beast.


_By Teddy Akiki & Claude (only one of us actually uses SLURM)_
