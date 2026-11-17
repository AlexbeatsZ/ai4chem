# Goal

Maintain reproducible Transition1x benchmarks and small-data fine-tuning on
OMEN WSL CUDA. v0.3.0 completed validation selection and final evaluation.

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
- v0.2.0 reads ten test reactions selected with seed 42 from the pinned JMP
  Hugging Face export. `data/transition1x_test_10.h5` retains 13,348 source
  configurations and the benchmark evaluates 100 final-band images on CUDA float64.
- Three baselines passed. OFF23 / MH-1 omol / MH-0 rgd1 barrier MAEs are
  0.614896 / 0.311887 / 0.606168 eV. Detailed outputs are in `results/v0.2.0/`.
- Installed pyarrow 26.0.1 in the isolated environment. Lock now uses version
  pins, including packaging==26.3, and is Linux CUDA specific.
- v0.3.0 fixed 77 development / 20 validation / 210 final reactions. Four 8-epoch
  runs completed on nested 64/256 train frames, one frame per unique train reaction.
- Validation selected mh1_full_256. On 210 final reactions barrier MAE changed
  from 0.245429 to 0.131641 eV, force component RMSE from 0.251791 to 0.111649 eV/A.
- Source train data is shard 0 of 13. `configs/v0.3.0_partitions.json` fixes all IDs.
- `results/v0.3.0/` has reports and bootstrap intervals. `models/v0.3.0/` has models.

# Active Work

- v0.2.0 small-sample data reading and pretrained benchmarking are complete.
- v0.3.0 expanded baselines, fine-tuning and final evaluation are complete.
- The 210-reaction final set was opened after validation selection and has now
  been reported. Further method choices informed by these results require a new
  independent evaluation design.
- Official full HDF5 download returned HTTP 403. Current empirical input is a
  public export with float32 positions/forces, with source revision and precision
  recorded in the manifest. Numeric comparison against the official original is pending.
- Full train-pool coverage, multiple seeds, longer training, pretraining-overlap
  checks, model-optimized NEB and Mac execution remain future work.
- MH-1 optional TorchScript compiled export failed with an upstream Irreps type
  error. Standard .model serialization and loaded inference passed for all runs.

# Build / Run / Test

Run from this project directory inside WSL.

```bash
mamba run -n ai4chem python smoke_test.py
mamba run -n ai4chem python -m pip check
mamba run -n ai4chem python inspect_models.py
mamba run -n ai4chem python prepare_transition1x.py --count 10 --seed 42
mamba run -n ai4chem python benchmark_transition1x.py
mamba run -n ai4chem python -m unittest discover -s tests -v
mamba run -n ai4chem python prepare_research_data.py
mamba run -n ai4chem python run_finetuning.py
mamba run -n ai4chem python run_research_evaluation.py
```

The smoke test requires CUDA and writes `results/smoke_h2o.extxyz` and
`results/smoke_summary.json`. It checks the reloaded energy and forces.

# Durable Lessons

- When invoking WSL from PowerShell, write UTF-8 without BOM and LF shell scripts
  to the approved temporary directory and execute them through their `/mnt/c` path.
  PowerShell's stdin pipeline introduced a BOM and a trailing CR in this session.
- Transition1x reaction groups concatenate NEB optimization iterations. Recover
  final bands using first endpoints and final interiors, including the single-band case.
- Inspect run_train.py when checking E0 modes. MACE 0.3.17 handles `estimated`
  there, outside the lower-level E0 helper, and uses the first foundation head's E0s.
- Before remove_pt_head, set the process default dtype from the source model
  like the official select-head CLI. Missing this caused E0 estimation dtype failure.

# Docs Index

- `README.md` contains the reproducible environment setup and inference command.
- `docs/log/2026-11.md` records the initial CUDA validation and observed warnings.
- `docs/design/transition1x.md` covers provenance, final bands, metrics, heads and E0.
  Read before changing Transition1x preprocessing or benchmark semantics.
- `docs/design/finetuning.md` covers v0.3.0 partitions, training and selection ordering.
