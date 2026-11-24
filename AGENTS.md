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
- v0.1 CUDA/readiness and v0.2 ten-reaction benchmarks passed; per-host locks
  remain separate. v0.3 selected mh1_full_256: final210 barrier MAE0.131641eV,
  force RMSE0.111649eV/A. Old287 test reactions are already used.
- v0.4 endpoint searches and five selected Gaussian chains completed; conditional
  selected-five barrier MAE0.103424eV. See results/v0.4.0 and ts-search.md.
- v0.5.0 is complete/published November23,2026 with eight verified assets.
  All96 strategy units/83 unique searches and three Gaussian reactions passed
  final verification. Holdout12 original S0/S1/S2 saddle+IRC3/3/3, fine-tuned5/6/6;
  strict pipeline3/3/3 and5/5/5. Nine DFT units reuse three unique jobs.
  Do not restart completed coordinators from historical runtime snapshots.
- OMEN Windows Gaussian is C:\Users\Meta\AppData\Local\Programs\g16w\g16.exe;
  CLI uses input/output arguments. Project wrapper uses process-local environment.
- Standard .model serialization and reloaded inference passed; optional upstream
  MH1 TorchScript export fails and is not the accepted artifact.

# Active Work

- November24 user authorized >=18h queues on roommate/OMEN/Mac. Read
  `docs/design/local-compute-queue.md` and `configs/overnight-20261124.json`.
  Seven paired random/path repeats are reserved: seeds24–26 OMEN,27–30 Mac
  (full seeds20261124–30). Do not redistribute original member01–05 ZIPs.
  Roommate receives three AIMNet chains0255/3342/4757 then quarantined train-side
  force labels,24h actual-execution admission budget;2GB/fourthreads,4GiB free.
  Runtime is results/overnight-20261124; verify owners before starting/recovering.
  Mac r2 owner10062 has two actual E0 workers; r1 failed before any training
  and is retained as mac-infra-r1. OMEN native22604/WSL61077 waits on v0.9 lock.
  Roommate native13616 started0255 at16:13Z; fresh SSH saw g16/l1110 and
  chain child33396 after submission returned. New force labels remain quarantined.

- Mac now has Intel Gaussian 16 A.03 / GaussView 6.0.16 under
  `~/.local/opt/gaussian16`; signed `~/Applications/GaussView.app` supplies its
  process environment. Water wB97X/6-31G(d) Opt/Freq and GUI log display passed;
  GUI submission, TS/IRC and production concurrency remain unverified. See mac-qc.md.
- User requested OMEN validation of cloud deployment first, then a user-provided
  cloud server test, then five-member task distribution. Do not create/buy a server
  or dispatch to other humans/cloud yet. Three existing local nodes are separately
  authorized on November24. Read `docs/design/cloud-workers.md`.
- Native Linux Gaussian remains pending the user's next-day files. ML readiness now
  passed independently on OMEN: fresh locked environment, CUDA force difference,
  two epochs/reload and native Windows-owned unattended return. Mac loaded the
  returned model; see docs/reports/cloud-ml-readiness/README.md. Actual cloud
  acceptance and original Gaussian/group-dispatch gates remain pending.
- User wants unattended group work. Five self-contained paired random4096/path4096
  seed20261124–28 packages are ready in results/group-training/paired-sampling-r1;
  no overlap with seed16–18. November24 assigns original five pairs to local
  queues and adds two fixed pairs; no other-human/cloud dispatch.
  Linux entry is bash member-NN/start.sh. WSL must use the native Windows
  scripts/launch_training_bundle.ps1 owner; generic WSL detach is rejected.
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
- v0.7 four arms completed 40 actual epochs each and validation60 at 03:44Z
  November24 (Nov23 local). random4096 selected, barrier MAE0.125968657 eV and
  force RMSE0.090547554 eV/A; see `docs/reports/v0.7.0-screening/README.md`.
- v0.9 obtained the released v0.7 runner lock and started the first CUDA force-loss
  control at 03:45:56Z; read `docs/design/optimizer-study.md`.
  Final models/router still require freezing before prospective static/search/DFT.
- v0.8 uses 512 train reactions/shared labels/equal updates. Mac native parent2061
  has five completed CPU models and two remaining trainers; searches are terminal.
  The previous global trainer
  was adopted without restart. `configs/v0.8.0_execution.json` records measured
  nominal oversubscription and can change admission capacity without restarting workers.
  Read `docs/design/specialization.md`; finite development analyzer waits for all
  seven benchmarks using `configs/v0.8.0_analysis.json`, without opening holdouts.
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
- Roommate AIMNet2-rxn/rxn8694 Gaussian finished at 02:35Z November 24 (Nov 23
  local); Mac recovered all raw evidence and independently verified the full chain
  at 03:43Z. Electronic barrier 2.625419123 eV versus MLIP 2.044812811 eV; one
  post-hoc development candidate, not a population success-rate/accuracy claim.
  See `docs/reports/v0.7.2-diversity-dft/README.md`. Owner5232 is terminal; no rerun.
- v0.5 is complete/published and must not be restarted. v0.6 r2 is terminal:
  Gaussian rxn9456 full chain passed (barrier 4.677548685 eV); Mac same-candidate
  chain failed with a real negative-curvature endpoint and exhausted backward IRC.
  `docs/reports/v0.6.0-pilot.md` records this one post-hoc recovery, not a new cohort.
- Mac rxn0666 positive-control full chain and four engine tolerances passed.
- Host access/recovery lives in synchronized `~/.agents/ai4chem--adaptive-ts-search/AGENTS.md`.
  No live coordinators remain from v0.6. ROG availability was not accepted.
- Source float32 export equivalence and foundation-pretraining overlap are unresolved.

# Build / Run / Test

OMEN WSL uses existing Mamba ai4chem; Mac uses the appropriate existing Pixi
interpreter (decision-models includes sklearn for the complete regression suite).
Read the corresponding design before restarting a historical entrypoint.

```bash
mamba run -n ai4chem python smoke_test.py
mamba run -n ai4chem python -m pip check
mamba run -n ai4chem python -m unittest discover -s tests -v
python scripts/experiment_status.py
# New local queue: inspect results/overnight-20261124/*/state.json FIRST.
# Mac: environments/mac-mace/.pixi/envs/default/bin/python -u scripts/run_paired_queue.py --protocol configs/overnight-20261124.json --node mac
# OMEN native: powershell -File scripts/launch_paired_queue.ps1 -Protocol configs/overnight-20261124.json
```

Full historical prepare/search/DFT commands remain in README.md and their designs.
Smoke requires CUDA and checks reloaded water energy/forces. Do not relaunch a
completed or active study merely to inspect its status.

# Durable Lessons

- WSL nohup lost its child after SSH exit in the deployment control. Use a native
  Windows WMI owner holding foreground WSL. In Start-Process, keep WSL option
  tokens unquoted; quote only paths that need it (all-quoted options reached zsh).

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

- `docs/design/local-compute-queue.md` fixes the November24 three-host queues.
- `docs/design/cloud-workers.md` defines cloud bootstrap, local/cloud acceptance and group-dispatch gates.
- `docs/project-background.md` is the research and course entrypoint. Read before
  planning new work; it integrates motivation, related literature and priorities.
- `docs/design/model-improvement.md` fixes v0.7 data roles; `optimizer-study.md` fixes v0.9.
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
