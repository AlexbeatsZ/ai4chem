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

- v0.11 MoE pilot completed. External composition mixture improves forces but
  worsens barriers. Matched 49k residual adapters each trained64reactions/512frames,
  2epochs/256updates: singlehead barrier0.109724/force0.090517,sharedexperts
  0.110541/0.090429 vs512 baseline0.111033/0.090201. Neither passes force guards.
  No promotion/retry. Real force-gradients/reload/symmetry and109regressions passed;
  newGPU/DFT0. Read docs/design/soft-mixture.md and docs/reports/v0.11.0-moe.
  v0.11.0-moe-pilot release8258281 verified8assets,including1024standardweight.

# Active Work

- November26 user authorized useful tasks on ALLthreehosts throughUTC+8
  November27 08:00 (=2026-11-27T00:00Z). Read
  docs/design/overnight-until-0800.md and
  configs/three-host-until-0800-20261126.json. No existing science restart.
- OMEN originalgrowth67371 trains2048s2; s0/s1 completed20epochs, validation
  barrier0.106280/0.103625, force0.082694/0.082712. Bothweights recoveredMac;
  energy/forces/forceFD<=3.30e-7 passed. No best-repeat-seed selection.
  Native35116/WSL75451 TS-study waits growth lock/terminal:512x8frames/8epochs
  uniform vs normalizedTSx2; strictguardadmits2048x8/20epoch fullmodel.
  Native25432/WSL75777 acceptance waits normalTSterminal,then<=3hreservedGPU
  admission:16units on8othermetaendpoints,disjointfromMac; no newadmission00:00Z.
- Mac actualowner69899, workers69904/69905 each4threads: fromfrozen2048s0,
  8480publicreplayframes/20epochs completecontrol plus544frames/4epochs public
  vsGaussian5D adaptationpilot. Realtraining/initialvalidation/epochs observed.
  Strictcommonvalidationguardadmitsmatched8480frame/20epoch adaptation.
  Slotsreleasedbefore00:00Zcanrun16fixed512/2048metaendpointsearchunits.
  After00:00Znonewstages; existingexactworkertreesnice>=10/naturallydrain.
  State/results:results/until-0800-20261126. Task-onlycaffeinate69900.
- Roommate old29700/39768 terminal; calibration30184 completed16/16,875.95s,
  all3Nforce/geometryalignmentpassed.5D reduces6D-public discrepancy but residual
  numericdifferencesremain. OldZIP34868884bytes/13591entries recoveredCRCpassed.
  Native16128 runs frozenoffpath32batch:2885Dframes=8reusedrefs+280newjobs,
  serial4threads/2GB/4GiBgate/max8nativehours; collector50081return+alignment.
  Newnative3248 holds1152unique5Dreserve(next128oldtrainreactions),waitsnormal
  firstbatchdrain+actualMacpilotgate. Gateonlypermitsonefinite dataextension,
  notmodelpromotion. Deadline00:00Zthen drain,3failurestop/no retries.
  Collector73508 observesnewreserve12h. No automaticmergedataset/full512expansion.
  Readconfigs/offpath32-20261126.json andoffpath128-until-0800-20261126.json.
- User prioritizes better deliverable models AND methods beyond scale. Starting
  OMENGPUallocation60%completeapplication/repeats,<=25%exploration,~15%acceptance;
  debitcommittedqueuefirst. Readtraining-methods.md,model-growth.md,ts-weight-study.md.
  v0.8sevenmodels/v0.9fivecontrols complete; global512 remainsfrozenMoEbaseline
  (validationbarrier0.111033270,force0.090200620,TS0.133513). v0.10expandsreaction
  coverage,notparameters. TS/relative-energy/AIMNet/EMA/LoRA areseparatemethods.
- November24pairedqueues terminalonMac; OMENgrowthfollowscompletedoldqueue.
  Seeds24–26OMEN/27–30Mac remainassigned, member01–05ZIPs mustnotredistribute.
  Roommate0255/3342 DFTverified,4757 failedfixedforwardIRC320; no scienceretry.
  v0.6terminal;9456Gaussianpassed/Macfailed,0666positivecontrolpassed.
- v0.7 roles remain256train/60validation/128meta plusprospective64train+64val.
  Allprospective labels stayunopened untilfinalmodel/evaluationbudgetfreeze.
  Old287testreactions alreadyused; nevercallunseen orfitnewrouteronthem.
  Source13shards requirecompletecontiguousfid histories; baseline4096labels retained.
  v0.7random4096 selected; v0.8specialists/router andv0.11MoE didnotjointlybeat
  strongbaseline. Routerkeepsonemodelperreaction; noimagewisehardswitching.
- Cloudacceptance andotherhuman dispatch remainseparategates. Do notcreate/buy
  serverorredistributereservedmemberbundles. Existinglocal3nodesauthorized.
  OMENMLreadiness passed; nativeLinuxGaussian/userprovidedcloudunaccepted.
  Mac Gaussian16A.03/GaussView6.0.16 installed/signed; waterOpt/Freq/logdisplay
  passed,GUI submission/TS/IRC/concurrencyunverified. Seecloud-workers.md/mac-qc.md.
- Hostaccess/recovery: synchronized~/.agents/ai4chem--adaptive-ts-search/AGENTS.md.
  Actualprivate receipts:results/coordination/compute-20261126 and
  results/until-0800-20261126; ownersandscientificcompletionaredistinct.
  Foundationpretrainingoverlap andpublicfloat32/originalHDF5equivalenceunresolved.

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

- `docs/design/overnight-until-0800.md`: three-host deadline, actual adaptation and gated label continuation.
- `docs/design/ts-weight-study.md`: finite TS objective pilot/application and DFT label calibration.
- `docs/design/soft-mixture.md`: external gates and matched shared-trunk residual pilot.
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
