# Goal

Maintain reproducible Transition1x benchmarks, fine-tuning and endpoint-driven
transition-state searches on OMEN WSL CUDA. v0.5.0 compares bounded multi-start
and failure-aware recovery using the existing CI-NEB and DFT validation workflow.

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
- Remote is `https://github.com/AlexbeatsZ/ai4chem`; GitHub reported public on 2026-11-22.
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
- v0.4.0 completed all twenty endpoint-only searches. Original MH-1 passes 7/10,
  fine-tuned MH-1 passes 6/10 under NEB, stationary-point, frequency and IRC checks.
- Existing Windows Gaussian is `C:\Users\Meta\AppData\Local\Programs\g16w\g16.exe`.
  Its CLI takes input/output file arguments. WSL calls the checked-in PowerShell
  wrapper, with a per-result scratch directory and process-local environment.
- Installed Sella 2.6.0 and CPU JAX 0.12.0.1 only in the ai4chem Mamba environment.
- Five selected fine-tuned candidates passed Gaussian TS/frequency, two-way IRC,
  stable-minimum and mapped-endpoint checks. On these selected five, electronic
  barrier MAE is 0.197624 eV for original MH-1 and 0.103424 eV for fine-tuned MH-1.
- rxn1152 IRC reached the 160-point budget, then checkpoint continuation detected
  a minimum at point 161. Gaussian's subsequent PESDtA Invalid IPoint error was
  recovered using a separate normally terminated endpoint optimization/frequency
  and geometry validation. normal_termination=false and recovery flags are retained.
- `results/v0.4.0/` contains all 20 searches, five DFT records, plots, report and
  verification.json. 31 regression tests, pip check and CUDA smoke test passed.
- MH-1 optional TorchScript compiled export failed upstream in v0.3.0. Standard
  .model serialization and loaded inference passed for every experiment.
- v0.5.0 fixes four development and twelve project holdout reactions, excluding
  all prior project reaction IDs. Train export shards 0 and 1 provide 223 complete
  unused endpoint pairs, including four formation-class cases.
- All 24 development strategy units completed with 25 unique attempts. Verified
  saddles for original S0/S1/S2 are 0/4, 1/4, 0/4 and fine-tuned are 0/4, 1/4, 1/4.
- configs/v0.5.0_frozen.json was written after pilot verification and before any
  holdout search. The original draft's recovery rules remain fixed for evaluation.
- Two rxn1961 baseline regressions passed. The shared-endpoint S0
  entrypoint matched the old barrier within 1.1e-10 eV and frequencies within
  1.1e-6 cm^-1. Pilot verification also checked actual perturbations and continuation.
- All 96 v0.5.0 strategy units now finished with 83 unique attempts. On the 12
  holdout reactions, original S0/S1/S2 saddle+IRC counts are 3/3/3; fine-tuned are
  5/6/6. Strict pipeline counts are 3/3/3 and 5/5/5 respectively. All 49 tests passed.
- At 2026-11-23T09:29:56Z, rxn0666 and rxn2737 completed and passed full DFT
  validation. rxn8204 is still in forward IRC (point 61 observed); final DFT
  acceptance and v0.5 release remain pending. Check OMEN for live state.

# Active Work

- Multi-device handoff starts at `docs/coordination/README.md`. OMEN remains the
  sole writer of this frozen CUDA experiment; Mac handles CPU reporting and ROG
  audits completed result snapshots. Each host uses its standard ai4chem checkout.
- `scripts/continue_v05.py` attaches to the existing OMEN search before running
  the unchanged DFT/verification pipeline. Its runtime state, PIDs and stage logs
  are under `results/coordination/`; inspect them before restarting any process.
- The user can arrange group-member compute for large independent batches with
  roughly one-day turnaround. Prepare concrete versioned task packages first;
  no group-member jobs have been sent or launched in this experiment.
- Independent Gaussian reaction jobs can run on different group members' hosts.
  The frozen OMEN CUDA identity concerns the MACE comparison, not a requirement
  that DFT runs on OMEN. Keep scientific inputs/settings and acceptance fixed,
  record execution provenance, and deduplicate by candidate identity. Prefer
  assigning complete TS/Freq -> IRC -> minimum/Freq reaction chains.
- The linked ChatGPT draft was read in native Chrome on 2026-11-22. Read
  docs/design/v0.5.0-protocol.md before changing controller, cache or new selection.
- Finish the preselected Gaussian subset rxn2737, rxn8204 and rxn0666. Preserve
  all nine strategy units and deduplicate identical TS coordinates. Complete report,
  verification and release after all native calculations reach terminal outcomes.
- The 210-reaction final set was opened after validation selection and has now
  been reported. Further method choices informed by these results require a new
  independent evaluation design.
- Official full HDF5 download returned HTTP 403. Current empirical input is a
  public export with float32 positions/forces, with source revision and precision
  recorded in the manifest. Numeric comparison against the official original is pending.
- Full train-pool coverage, multiple seeds, longer training, pretraining-overlap
  checks and Mac execution remain future work.
- Expanded independent search evaluation and multiple path initializations remain
  future work. v0.4.0 uses ten exposed reactions and one IDPP initialization each.

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
OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu mamba run -n ai4chem python ts_search.py
mamba run -n ai4chem python validate_ts_dft.py
mamba run -n ai4chem python summarize_ts_search.py
mamba run -n ai4chem python prepare_v05_reactions.py
mamba run -n ai4chem python diagnose_ts_search.py
mamba run -n ai4chem python adaptive_ts_search.py --role development
mamba run -n ai4chem python adaptive_ts_search.py --freeze
mamba run -n ai4chem python adaptive_ts_search.py --role holdout
mamba run -n ai4chem python validate_ts_dft_v05.py
mamba run -n ai4chem python summarize_ts_search_v05.py
python scripts/experiment_status.py
# Only on OMEN WSL, after checking existing coordinator state:
# Foreground WSL: bash scripts/run_v05_omen.sh
# Durable background: PowerShell scripts/launch_v05_omen.ps1 -Mode coordinator
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
- ASE 3.29's Gaussian frequency-log reader can attach a force array with fewer rows
  than atoms. DFT validation exports checked coordinates and separately parsed SCF
  energy, omitting those unusable parsed forces.
- Compare separated molecular products using mapped bonds and rigid alignment per
  connected component. Whole-system RMSD conflates fragment arrangement with chemistry.
- The H-H geometric bond cutoff is 0.744 A at the current radius multiplier.
  rxn8080 crossed it after a 0.002 A H-H distance change. Inspect mapped pair
  distances before interpreting an endpoint graph mismatch chemically.

# Docs Index

- `README.md` contains the reproducible environment setup and inference command.
- `docs/log/2026-11.md` records the initial CUDA validation and observed warnings.
- `docs/design/transition1x.md` covers provenance, final bands, metrics, heads and E0.
  Read before changing Transition1x preprocessing or benchmark semantics.
- `docs/design/finetuning.md` covers v0.3.0 partitions, training and selection ordering.
- `docs/design/ts-search.md` covers endpoint-only initialization, saddle/IRC checks,
  Gaussian integration and numerical acceptance thresholds. Read before changing searches.
- `docs/design/v0.5.0-protocol.md` fixes the failure-aware search experiment,
  endpoint-only project holdouts, shared costs, cache identity and evaluation rules.
- `docs/coordination/README.md` is the cross-device task board and recovery entry;
  node records distinguish installed environments and verification scope.
