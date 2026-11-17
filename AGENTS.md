# Goal

Run MACE-OFF23-small on OMEN's RTX 4060 Laptop GPU in WSL Ubuntu 24.04.
Calculate water's potential energy and three atomic forces, save the results,
and verify that the saved structure can be read back.

# Current State

- Project directory is the Windows checkout mounted at
  `/mnt/c/Users/Meta/Project/Workspaces/ai4chem` in WSL.
- OMEN GPU was detected with 8188 MiB VRAM on 2026-11-16.
- Existing Mamba is `/home/meta/.local/opt/miniforge3/bin/mamba`, version 2.6.0.
- Use the isolated `ai4chem` Mamba environment with Python 3.12.
- v0.1.0 validated with Python 3.12.16, torch 2.15.0+cu134, mace-torch
  0.3.17 and ASE 3.29.0. Initial download and cached rerun both passed.
- Water energy is -2081.1162109375 eV with finite 3x3 forces. Saved ASE
  results passed readback checks. Full outputs are under `results/`.
- Private remote is `https://github.com/AlexbeatsZ/ai4chem`.

# Active Work

- Local CUDA inference validation is complete.
- Training, Transition1x, Mac execution and reaction-path accuracy are future work.

# Build / Run / Test

Run from this project directory inside WSL.

```bash
mamba run -n ai4chem python smoke_test.py
mamba run -n ai4chem python -m pip check
```

The smoke test requires CUDA and writes `results/smoke_h2o.extxyz` and
`results/smoke_summary.json`. It checks the reloaded energy and forces.

# Durable Lessons

- When invoking WSL from PowerShell, write UTF-8 without BOM and LF shell scripts
  to the approved temporary directory and execute them through their `/mnt/c` path.
  PowerShell's stdin pipeline introduced a BOM and a trailing CR in this session.

# Docs Index

- `README.md` contains the reproducible environment setup and inference command.
- `docs/log/2026-11.md` records the initial CUDA validation and observed warnings.
