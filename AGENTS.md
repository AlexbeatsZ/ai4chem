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

- v0.14.0-four-hour-compute prerelease published frome3f654a:5assets verified,
  sourceZIP downloadCRC/config/entrypoint match and2048s2actualCPUreload passed.
  Includes completed thirdrepeatweight; EMA/DFT stillrunning, not completedclaims.

# Active Work

- 2026-11-28 02:16Z research continuation (user explicitly authorized more bounded
  compute): OMEN phase-1 128 existing meta-development DFT paths for both
  MACE2048/4096 complete (first script stopped after writing metrics due missing
  sklearn; distinct recovery-only analysis completed; no recomputation).
  2048/4096 average fixed-path tangent force RMSE .0639026/.0849290
  eV/A, energy-slope RMSE .0863794/.1197209 eV/A; 94/128 tangent
  errors increased. Existing 16 paired searches 2048 5 strict vs4096 3;
  no novel DFT was run for the phase-1 static study.
- OMEN Popcornn official repository pinned e4a02f4c in isolated
  /home/meta/Project/Workspaces/popcornn/.venv, using the existing
  ai4chem PyTorch environment's packages, separate optional dependencies,
  custom small-molecule pure-Torch radius graph; no shared-env package
  replacement. Two-geometry MACE parity passed (<1e-9eV and <1e-8eV/A).
  The upstream integrate_gradient=True vmap path fails on scripted MACE;
  integrate_gradient=False passed 2+2 step smoke. Active bounded
  results/popcornn-compare-20261128 owner93038: 2 models×8 previously
  examined meta reactions, same endpoints, each candidate undergoes Sella,
  Hessian, IRC and minimum stability checks; 2100s/unit, four positive-control
  pilot units gated before expanding, no retries, new admission ends
  2026-11-28T19:00Z. Automatic paired comparison aggregator running.
- Mac results/failure-policy-20261128 completed formula-disjoint 5-fold
  offline post-S0 decision policy; small n=32 and S1/S2 costs modeled as
  independent FULL restarts after S0, not observed sequential computation.
  At lambda=0, ridge 10 successes/18433s versus fixed S1 10/25095s
  and stop 8/6014s; hypothesis only, no independent prospective claim.
- Roommate PC (authorized again) owner35152:
  results/frontier-dft-rescue-20261128 2 frozen Gaussian full-chain
  candidates rxn3849 and rxn9706, both strict MACE2048 successes
  where MACE4096 failed, meta-development only. 4threads/2GB and
  4GiB free-memory gate, last observed waiting at3.9GiB; admission
  until 2026-11-28T14:15Z, no retries. Mac collector75920 observes
  raw evidence and independent assessor76046 validates Gaussian TS,
  frequencies, both IRC branches and mapped endpoints once returned.
  Do not imply DFT started until queue state says so; no training label merge.
- Research priority: static tangent/slope and search failure mechanism,
  then algorithm-matched Popcornn vs CI-NEB success/cost; select new
  DFT for genuine novel rescues only with independent guards.
  Original prospective holdout remains unopened. Stop before launching
  any extra uncapped GPU or Gaussian tasks without a fresh protocol.

- November27 13:19Z: user requests a finite overnight batch through
  November28 09:00 Asia/Shanghai (= November28 01:00Z), preserving useful
  science and existing work. Local task states in git-ignored results; no new
  automatically purchased compute; roommate 4GiB gate unchanged. A scheduled
  one-time 09:00 Beijing verification was created separately in ChatGPT.
  Mac decision-surrogate-20261127 owner87890 actively trains three matched
  frozen-backbone adapter objectives on 64 reactions x8 frames, 4 epochs:
  standard E/F, reaction relative energy, differentiable soft barrier (tau0.1eV).
  THIS IS NOT end-to-end DFL through NEB, only a downstream soft surrogate;
  validation60 only, hard stop by 01:00Z or4h. Next Mac
  active-query-router-20261127 supervisor88781 waits for first terminal,
  then balanced random vs uncertainty vs decision-ambiguity adaptive sparse
  expert queries, 3 seeds/four fractions, 5-fold grouped OOF, <=3h.
  Mac decision-surrogate-2048-20261127 supervisor91161 waits and runs
  larger 128-reaction,6-epoch two-arm matched pilot ONLY if first objective
  jointly beats matched control and frozen baseline; otherwise stops.
  OMEN original4096x4 training and guarded8-search owner81732 still live.
  New meta128 validation supervisor84150 waits for both, evaluates 4096
  vs development metadata, and searches old8 only if former guard skipped.
  Additional OMEN supervisor84201 already froze16 new meta endpoints <=16
  atoms with endpoint-only selection, and waits for previous acceptance,
  then runs paired 2048/4096 actual MLIP search up to3h per model, no retries.
  Roommate queue frontier-dft-20261127 owner16876 admitted four paired
  AIMNet2/MACE TS/Freq/IRC chains but WAITING_FOR_4GIB_FREE, no Gaussian
  started at 13:18Z. 09:00 extension owner39340 stays idle until prior
  queue terminal, at most copies unattempted jobs into a fresh queue with
  01:00Z admission. Does not duplicate begun/failed jobs. Mac collectors
  82099 and89440 watch original and extension, respectively. No change to
  source validation split or sealed prospective holdout. Jobs stop on real
  failure/resource gate; not a promise all GPUs/hosts stay busy until 09:00.
  This block supersedes older fixed 07:01 admission notes below.

- November27 13:02Z: user confirms roommate capacity available again. OMEN
  4096x4 training logged epoch18 at12:55Z, GPU79%; conditional acceptance
  owner81732 still waiting. Mac all48 basic and31 advanced routing arms
  complete; TabPFN sparse 12.5/25/50/100% MAE .159540/.154418/.145471/.137624,
  full fixed expert .131575; no promotion. Matched AIMNet2 searches 2/8,
  MACE512 and2048 each3/8, AIMNet successes subset of both.
  Roommate previous chain terminal, new frontier-dft-20261127 owner16876
  deployed four paired DFT chains (rxn0813/rxn0673 AIMNet2 vs2048 MACE),
  serial4threads/2GB, 4GiB free gate, admission ends 23:01Z; actual status
  waiting_for_4GiB_free with2.628GiB available, zero Gaussian processes.
  Initial memory preflight rejected; second extraction succeeded but script
  execution policy blocked before owner creation; process-local bypass then
  launched exactly once. No global security policy change. Mac collector82099
  observes up to24h and recovery82240 observes old0255 up to2h, no science
  retry. New four chains are post-hoc development DFT acceptance; do not merge
  labels, do not open prospective. Deadline is admission not a hard runtime cap.

- November27 follow-up (07:05Z): user says roommate has left and computer is
  offline; count its capacity as zero, do not restart, reconnect or dispatch
  additional jobs. Existing collectors/evidence only. User requests larger
  bounded batches without repeated manual handoff. OMEN4096x4 remains active.
  Native53640/WSL81732 follow-up waits <=18h for completed joint-validation
  eligibility; if eligible, run matched eight meta endpoints with CUDA, <=3h
  admission, no retry or holdout; otherwise stop. Local ignored state:
  results/frontier-followup-20261127. Device differences affect timing.
  Mac frontier-pilots-20261127: retrospective 128x8 sparse router
  48 arms completed, formula-disjoint five folds; 32 legacy search-unit
  recovery policy OOF and 83 unique attempts DFL data audit completed.
  Additional fixed-mask TabPFN/TabICL and direct expected-cost/CE routing
  owner15424 launched, finite arms; AIMNet2 matched8 endpoint search owner14806
  actual first reaction started, <=3h admission, no DFT. These are decision-layer
  experiments, not DFL PES training. Stop after the bounded batch, do not
  autoexpand conventional methods or expose prospective labels.

- November27 research-priority update (07:01Z): user authorizes selective pause
  and says existing coursework evidence is sufficient. Preserve the running
  4096x4 equal-label experiment and previously admitted Gaussian TS verification;
  do not automatically expand conventional training/DFT or open prospective.
  Mac active-learning candidate preparation completed (16 reactions, 560
  candidates, 60 unique DFT inputs). Dispatch owner95806 terminated before
  archive creation/remote submission; state paused_by_research_priority.
  Retain inputs, no new active-learning DFT until a new bounded admission.
  Roommate latest local evidence: both rxn8694 chains assessed, rxn0255 started;
  fresh relay hostname resolution failed, no assertion of live progress.
  Next new research protocols prioritize sparse-supervision routing,
  learned failure-mode-aware intervention and decision-focused PES training.
  Literature claims require primary-source verification; conventional
  active learning and committee variance are baselines, not novelty claims.
  These new methods are not yet implemented or authorized as unbounded queues.
  This update supersedes earlier automatic continuation instructions below.

- November27 next compute: user authorized Mac/OMEN/existing roommate. OMEN native
  owner26044 / WSL81448 runs `run_equal_budget_growth.py` once, from original
  MH1, seed20261116, 4096 reactions x4 labels x20 epochs/batch4 =81920 updates.
  Nested reaction identities; same label count as2048x8, not identical frame
  selection. Output results/equal-budget-20261127; development validation only.
- Mac completed three-seed analysis of288 quarantined DFT frames /128 pairs:
  Spearman0.510652, top-quartile force error0.265724 vs random0.198212eV/A;
  direction HVP RMSE means1.130644/1.067148/1.072549eV/A². No training merge.
  Mac95088 prepares16 train-side reactions/short NEB trajectories,32 high
  disagreement and32 random arm slots (overlap reused), no prospective data.
  Mac95806 gates a single native serial DFT batch after the old queue and
  roommate40712 (unadmitted rxn0255 candidate) drain; admission ends
  2026-11-27T18:32:50Z, 4 job-hour cap, no retry. Actual state paths
  results/active-learning-pilot-20261127 and results/active-learning-dispatch-20261127.
  Raw returns must be aligned, and matched training protocol frozen before merge.
  Roommate previous IRC stays intact. Source relay status verified; active pilot
  transfer/launch not yet exercised. Local transport config stays git-ignored.

- Latest user explicitly requested another ~4h on all three nodes after the old
  November27 08:00UTC+8 cutoff. New admission deadline is November27 12:20UTC+8
  (=2026-11-27T04:20Z). Read docs/design/four-hour-continuation.md and
  configs/four-hours-20261127.json. Old terminal evidence is preserved.
- OMEN v0.10 all four models complete, including2048s2/20epochs/best14:
  barrier0.107642/force0.083162/TS0.123912. All2048seeds mean barrier0.105849,
  sampleSD0.002043. Frozen2048s0 remains baseline; no best-repeat selection.
  TS8epoch control/weighted barrier0.114585/0.121462; weighted rejects promotion.
  Old16unit acceptance failed at model loading before science (missing512weight),
  now restored56945307bytes. Do not restart these old owners.
- New OMEN native4932/WSL Python79004 runs one full8480frame/20epoch EMA0.99
  candidate from2048s0, matched to currentMacpubliccontrol seed2026112601,
  lr1e-5/E0foundation; actual8frameEMA save/reload/forceFD3.05e-7 passed.
  State:results/four-hours-20261127/omen. Initial status-format startup failure
  preserved as startup-failed-state.json before any scientific training.
  One fullcandidate/no grid/retry; admitted20epochs naturally drain after04:20Z.
- Mac original69899/fullworker69904/train69962 still completing20epochs;
  at00:25Z epoch18 complete, nice10. Adaptation69905 terminal: publicpilot
  barrier0.106378/force0.082435/TS0.123516, Gaussian5Dpilot
  0.110124/0.088032/0.129886. Both fullpromotion and128label extension denied.
  Existing original288labels are complete, aligned and quarantined; no expansion.
- New Mac owner95219/search95263+95264 each3threads,nice10 runs previously
  unexecuted OMEN frozen8metaendpoints,16units in results/four-hours-20261127/search;
  <=3h admission per worker, absolute04:20Z cutoff, currentunit drains.
  Oldother8meta searches complete:512and2048 each4/8strictMLIP, differentsets;
  these are development/noDFT claims. Task caffeinate95220 plus original69900.
- Roommate original32batch16128 finished280/280newjobs,288totalwith8reusedrefs,
  8084.91jobseconds; Maccollector50081 alignedcomplete. Reserve3248 terminal
  pilot_gate_declined,0jobs; collector73508 recovered rawinputevidence. No retries.
- Roommate new serialowner29228 at results/four-hours-20261127/roommate:
  three eligible newmodelDFTchains(2048/8694,512/8694,2048/0255), from smallest
  three oldmeta reactions/sixplannedunits; rejectedMLIPunits retained.4threads/2GB,
  4GiBfreegate,04:20Z newchain cutoff, boundedadmittedchain naturally drains.
  Actualg16/l1110observed. Macindependentcollectors97710/97711/97712 use verified
  per-commandhost/keyoverride,8h finiteobservations; no science restart or proxychange.
- Modelcollector7027 is active with finite8h read-only return and assessment.
- Source records private results/coordination/compute-20261126/{growth-final-state,
  ts-weight-final-state}.json and results/four-hours-20261127. Modelcollector
  scripts/collect_four_hour_model.py returnsactualEMAweight, CPUforceFD, common
  completecontrol+2048s0metrics before any eligibility decision. Prospective closed.
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

- `docs/design/four-hour-continuation.md`: latest four-hour deadline, EMA full candidate and independent acceptance.
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
