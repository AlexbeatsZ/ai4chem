# Goal

Complete the Artificial Intelligence in Chemistry course research project on
small-data MACE fine-tuning and failure-aware adaptive transition-state search.
Read `docs/project-background.md` first for course requirements, related papers,
the two research levels, measured evidence and actual deliverables.
Study how model accuracy and bounded search recovery affect verified TS success
and compute cost, using endpoint-driven CI-NEB, frequency, IRC and independent DFT.
v0.5.0 is released. The separate v0.6.0 pilot checks the sole extra recovery and
calibrates native Mac DFT. On November 23 the user clarified that they will write
the report later and that roughly two weeks should be used to improve the model.
The active priority is controlled data/optimization studies, repeated training,
prospective evaluation and targeted DFT labels where they improve the potential.
For long calculations, stay active through waits (including overnight), inspect
completion and advance the next authorized stage; keep handoff logs current.
Reported course presentation dates are December 7 and 14, 2026. Original course
instructions still determine the group's assigned slot and submission details.

# Current State

- OMEN checkout is `C:\Users\Meta\source\repos\ai4chem`, mounted at
  `/mnt/c/Users/Meta/source/repos/ai4chem` in WSL. Mac checkout is
  `/Users/meta/Developer/ai4chem`. Both were externally migrated on November 23.
- OMEN RTX4060 has 8188 MiB VRAM; isolated `ai4chem` uses Python 3.12 and
  Mamba 2.6.0 at `/home/meta/.local/opt/miniforge3/bin/mamba`.
- v0.1.0 validated with Python 3.12.16, torch 2.15.0+cu134, mace-torch
  0.3.17 and ASE 3.29.0. Initial download and cached rerun both passed.
- v0.1 water finite energy/forces and saved ASE readback passed; outputs in `results/`.
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
- At 2026-11-23T10:56:30Z, all three Gaussian reactions completed and passed full
  TS/frequency/IRC/minimum validation. Nine strategy units reuse these three jobs.
  Final verification.json passed. v0.5.0 was published at 2026-11-23T11:10:37Z;
  all eight assets and ZIP sizes were verified. Coordinator ready_for_review is a
  retained runtime snapshot; release completion is recorded here.

# Active Work

- User has about two weeks and wants better models; they will write the report.
- Read `docs/design/model-improvement.md` and `decision-routing.md` before changes.
- v0.7 freezes 256 MACE train reactions, 60 validation, 128 new meta-development,
  and prospective holdouts of 64 train + 64 val reactions. Concrete IDs/endpoint
  records are in `configs/v0.7.0_partitions.json`, frozen before new training.
- No prospective holdout labels are opened by the screening runner. The old 287
  test reactions are already used; never call them unseen or fit the new router on them.
- All 13 pinned train shards are downloaded; source files are frame-sharded. The first
  shard has portions of 9,561 reactions and incomplete histories for all old 256.
  Final paths require complete contiguous fid histories across the full source.
- First-stage arms: anchor256_long, path2048, random4096, path4096; same MH-1
  foundation, up to 40 epochs/patience 12, original loss/lr. All retain old anchors.
  Data sampling contrasts have equal reaction counts; only the two 4096 arms have
  equal label/epoch ceilings. Record actual time and updates.
- Next: optimizer/replay comparisons, three seeds for promising settings, freeze
  models/router, then prospective static/path-search/DFT validation.
- v0.8 specialization uses shared labels/equal updates; read `docs/design/specialization.md`.
- User proposed an expert library with a local decision router. First quantify
  oracle headroom on meta-development, then compare endpoint-only TabPFN/TabICL,
  Jev-style local decisions and simple baselines with grouped out-of-fold estimates.
  Fix one expert for an entire reaction path; image-by-image switching is not built.
- OMEN: `results/v0.7.0/state.json`, official logs under training/, launcher records
  under `results/coordination/model-v07/`. Check actual owner before restarting.
- Mac: isolated `environments/decision-models` for local MLX/MPS inference.
  Package installation, real weight inference and chemistry benefit are separate checks.
- Local TabPFN/TabICL MPS and jevmlx MLX inference passed. The 128-reaction static
  meta pilot shows no force-safe oracle gain with the old expert pool; see
  `docs/reports/v0.7.0-routing/README.md`. Endpoint-search benefit is still untested.
- Eight experts/four families now evaluated: AIMNet2/rxn, ANI and Orb were added.
  Read `docs/design/expert-diversity.md`; the new grouped pilot shows no deployed
  multi-metric routing gain. Eight-reaction/three-expert endpoint searches are frozen.
- v0.5 is complete/published and must not be restarted. v0.6 r2 is terminal:
  Gaussian rxn9456 full chain passed (barrier 4.677548685 eV); Mac same-candidate
  chain failed with a real negative-curvature endpoint and exhausted backward IRC.
  `docs/reports/v0.6.0-pilot.md` records this one post-hoc recovery, not a new cohort.
- Mac rxn0666 positive-control full chain and four engine tolerances passed.
- Host access/recovery lives in synchronized `~/.agents/ai4chem--adaptive-ts-search/AGENTS.md`.
  No live coordinators remain from v0.6. ROG availability was not accepted.
- Source float32 export equivalence and foundation-pretraining overlap are unresolved.

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

- MACE 0.3.17 non-distributed patience logs "stopping" but its exit flag is None;
  check actual termination and logged epochs. v0.7 retains its 40-epoch ceiling.
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
- Gaussian 6-31G(d) used 6D while PySCF defaults to spherical functions. For the
  CHNO/d-only check, cart=True restored matching basis counts; the same basis
  label alone did not guarantee comparable energies. Higher shells need explicit care.

# Docs Index

- `docs/project-background.md` is the research and course entrypoint. Read before
  planning new work; it integrates motivation, related literature and priorities.
- `docs/design/model-improvement.md` fixes v0.7 training, data roles and evaluation.
- `docs/design/decision-routing.md` and `expert-diversity.md` fix routing and expanded-family studies.
- `docs/design/v0.6.0-pilot.md` fixes the post-hoc candidate, Mac control, budgets
  and acceptance before either new full-chain computation starts.

- `README.md` contains the reproducible environment setup and inference command.
- `docs/log/2026-11.md` records the initial CUDA validation and observed warnings.
- `docs/design/transition1x.md` covers provenance, final bands, metrics, heads and E0.
  Read before changing Transition1x preprocessing or benchmark semantics.
- `docs/design/finetuning.md` covers v0.3.0 partitions, training and selection ordering.
- `docs/design/ts-search.md` covers endpoint-only initialization, saddle/IRC checks,
  Gaussian integration and numerical acceptance thresholds. Read before changing searches.
- `docs/design/v0.5.0-protocol.md` fixes the failure-aware search experiment,
  endpoint-only project holdouts, shared costs, cache identity and evaluation rules.
- `docs/design/mac-qc.md` defines the native Mac PySCF checks, concurrency,
  cross-engine basis conventions and separate Gaussian installation requirement.
- `docs/coordination/README.md` is the cross-device task board and recovery entry;
  node records distinguish installed environments and verification scope.
- `docs/coordination/roommate-gaussian.md` records the added Gaussian worker,
  tested command, resource limits and remaining background-launch acceptance.
