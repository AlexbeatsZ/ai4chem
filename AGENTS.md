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
The active priority is stronger deliverable models plus a defensible method
improvement. Cap exploratory comparisons and reserve compute to apply useful
methods, train/repeat full weights and independently verify the selected model.
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
- v0.1–v0.4 CUDA, training, searches and selected Gaussian chains are complete.
  v0.3 selected mh1_full_256; old287 test reactions are already used.
- v0.5 is published/verified:96 strategy units,83 unique searches,3 Gaussian jobs.
  Holdout12 saddle+IRC original3/3/3, fine-tuned5/6/6; strict3/3/3 and5/5/5.
  Detailed metrics remain in version reports. Never restart completed coordinators.
- OMEN Windows Gaussian is C:\Users\Meta\AppData\Local\Programs\g16w\g16.exe;
  CLI uses input/output arguments. Project wrapper uses process-local environment.
- Standard .model serialization and reloaded inference passed; optional upstream
  MH1 TorchScript export fails and is not the accepted artifact.

# Active Work

- November25 user reiterated: prioritize training better models for group results.
  They also require methods beyond scale; v0.10 expands data, not parameters.
  Read `docs/design/training-methods.md`: TS-aware objectives, common-DFT
  AIMNet adaptation, targeted training-side labels, then EMA/LoRA controls.
  November25 further clarified: try the candidate methods in bounded pilots,
  reserve compute for applying winners to actual models; initial GPU allocation
  60% complete training/application,<=25% exploration,~15% final acceptance.
  Debit already-committed queue cost first; do not launch every method at scale.
  v0.8 seven models/analysis are complete. Global512 validation60 barrier MAE
  0.111033270eV, force RMSE0.090200620eV/A; v0.9 completed all five new runs
  and retained lr1e-4/force1. Wider reaction coverage is the next training priority.
  Read `docs/design/model-growth.md`, `configs/v0.10.0.json`: nested1024/2048
  reactions,8frames each,20epochs; retain all512 old reactions/4096 exact labels.
  Two first-seed models then two fixed repeats only for a guarded improvement.
  OMEN data preparation passed10:41Z:8192/16384frames, all4096 old labels retained.
  Native52656/WSL31216/Python67371 is verified waiting on the old owner.lock;
  it will train after overnight/v09/v07 locks and terminal state. No restart.
  v0.10.0-model-growth prerelease published10:43Z against2bc4491; four assets
  verified, including the completed512-model weight (not an untrained growth model).
  Mac recovery: results/coordination/model-v10/omen-activation-verification.json.

- November24 user authorized >=18h queues on roommate/OMEN/Mac. Read
  `docs/design/local-compute-queue.md` and `configs/overnight-20261124.json`.
  Seven paired random/path repeats are reserved: seeds24–26 OMEN,27–30 Mac
  (full seeds20261124–30). Do not redistribute original member01–05 ZIPs.
  Roommate receives three AIMNet chains0255/3342/4757 then quarantined train-side
  force labels. November24 after idle-app cleanup user authorized two-slot pool:
  `configs/roommate-pool-20261124.json`,2GB/fourthreads each,6GiB launch gate,
  max2 workers,24h wall admission then drain; adopt current chain, no restart.
  Runtime is results/overnight-20261124; verify owners before starting/recovering.
  Mac37519 now has all four random arms completed and four path arms running;
  November25 07:00Z admission2/nice10 transition verified. Original science
  drains naturally; old scheduler10062 is retired. r1 remains mac-infra-r1.
  OMEN61077 completed seed24 and is training path25; seed26 remains pending.
  ROOMMATE RESUMED by new user instruction November25 12:13Z after11:16Z pause.
  Exact scheduler29700/conhost21336 restored after command/creation/thread checks;
  Mac collectors13752/13753/13755/16314 resumed, no replacement/restarted science.
  Fresh12:15:56Z saw activeGaussian/force log; terminal records173→178,free6.95GiB.
  Original17:24:51Z admission deadline unchanged; max2/fourthreads/2GB/6GiB gate.
  Receipts: worker queue/user-resume-20261125.json and Mac
  results/coordination/model-v10/*user-resume*. Labels remain quarantined.

- Mac Gaussian16A.03/GaussView6.0.16: ~/.local/opt/gaussian16, signed GUI entry.
  WaterOpt/Freq/log display passed; GUI submission/TS/IRC/concurrency unverified.
  See mac-qc.md for paths, signatures and process-local environment.
- User requested OMEN validation of cloud deployment first, then a user-provided
  cloud server test, then five-member task distribution. Do not create/buy a server
  or dispatch to other humans/cloud yet. Three existing local nodes are separately
  authorized on November24. Read `docs/design/cloud-workers.md`.
- Native Linux Gaussian/user-provided cloud acceptance remain pending. OMEN ML
  readiness passed isolated lock/CUDA force difference/twoepochs/reload/native
  unattended return; Mac reloaded it. See docs/reports/cloud-ml-readiness/README.md.
- Five self-contained group-training packages are now reserved to local queues;
  no other-human/cloud dispatch. Linux: bash member-NN/start.sh; WSL requires
  scripts/launch_training_bundle.ps1 native owner, generic detach is rejected.
- Read `docs/design/model-improvement.md` and `decision-routing.md` before changes.
- v0.7 freezes 256 MACE train reactions, 60 validation, 128 new meta-development,
  and prospective holdouts of 64 train + 64 val reactions. Concrete IDs/endpoint
  records are in `configs/v0.7.0_partitions.json`, frozen before new training.
- No prospective holdout labels are opened by the screening runner. The old 287
  test reactions are already used; never call them unseen or fit the new router on them.
- All13 pinned train shards are on OMEN; paths require complete contiguous fid
  histories across the full source. v0.7 four controlled arms retain old anchors;
  only random4096/path4096 share label/epoch ceilings. Details remain in design.
- v0.7 four arms completed 40 actual epochs each and validation60 at 03:44Z
  November24 (Nov23 local). random4096 selected, barrier MAE0.125968657 eV and
  force RMSE0.090547554 eV/A; see `docs/reports/v0.7.0-screening/README.md`.
- v0.9 finished November24 21:31Z; higher learning rate/force weight did not
  replace the baseline. Read `docs/design/optimizer-study.md`.
- v0.8 global512 is the best current validation candidate. Development analysis
  completed November24 18:41Z; specialists/router did not beat it consistently.
  Raw results: results/v0.8.0/development_analysis. All holdouts remain unopened.
- Expert library/router uses meta-development/groupedOOF; each reaction retains
  one expert. Image-by-image switching is not built. See decision-routing.md.
- OMEN per-version state/official training logs and results/coordination/model-v*/
  identify owners. Mac decision-models environment supports MLX/MPS analysis;
  installation, real inference and chemistry benefit are distinct acceptance gates.
- TabPFN/TabICL MPS and jevmlx MLX inference passed. Old-pool meta128 showed no
  force-safe oracle gain; see docs/reports/v0.7.0-routing/README.md.
- Eight experts/four families now evaluated: AIMNet2/rxn, ANI and Orb were added.
  Read `docs/design/expert-diversity.md`; the new grouped pilot shows no deployed
  multi-metric routing gain. All24 endpoint search units are complete: oldMACE
  strict1/8,AIMNet4/8,Orb0/8; MACE/AIMNet successes are disjoint. This does not
  compare the new512 model or establish a DFT population rate. Static specialist
  systems also lost to general512; see docs/reports/v0.8.0-specialization.
- Roommate AIMNet/rxn8694 fullGaussian independently passed: electronic barrier
  2.625419123eV vs MLIP2.044812811eV; one post-hoc candidate. Owner5232 terminal,
  no rerun. See docs/reports/v0.7.2-diversity-dft/README.md for raw evidence.
- v0.6 r2 is terminal: Gaussian9456 passed, Mac failed with negative endpoint
  curvature/exhausted backwardIRC. docs/reports/v0.6.0-pilot.md is post-hoc, not a cohort.
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
# Local queue: inspect results/overnight-20261124/*/state.json FIRST; no restart.
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

- `docs/design/model-growth.md`: v0.10 wider coverage and guarded seed repeats.
- `docs/design/training-methods.md`: beyond-scale directions and expert evidence.
- `docs/design/local-compute-queue.md` fixes the November24 three-host queues.
- `docs/design/cloud-workers.md`: bootstrap, acceptance and group-dispatch gates.
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
