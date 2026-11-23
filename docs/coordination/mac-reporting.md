# Mac CPU reporting and regression checks

- Task Key: ai4chem--v05-mac-reporting
- Host: Mac, `/Users/meta/Project/Workspaces/ai4chem`
- Last Verified: 2026-11-22 (America/Los_Angeles)
- Status: CPU snapshot report rebuilt; early DFT scheduling entrypoint ready for OMEN

## Scope and ownership

The Mac handles offline reporting and regression checks. OMEN remains the owner of
the frozen CUDA search matrix and Gaussian calculations. The Mac does not run a
second copy of the frozen search or change any model, selection or search setting.

The source checkout was cloned at `761e580`; the OMEN reporting changes to
`summarize_ts_search_v05.py` and `tests/test_v05_summary.py` were copied here before
this work and retained. Only the report's indirect dependency on the inference
module was removed: CSV writing now uses the same standard-library `csv.DictWriter`
settings directly. The five implementation files in the frozen identity were not
changed by this reporting task.

## Environment and reproducible commands

The isolated `.venv` uses Python 3.12.14 on macOS arm64. Direct analysis dependency
pins are in `requirements.analysis.txt`; the existing Linux CUDA lock is unchanged.
Neither PyTorch nor MACE is installed in this environment.

From the Mac project directory:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.analysis.txt
TMPDIR=/tmp/.agents .venv/bin/python -m unittest discover -s tests -v
TMPDIR=/tmp/.agents .venv/bin/python summarize_ts_search_v05.py --help
uv pip check --python .venv/bin/python
```

Create `/tmp/.agents` before using it as `TMPDIR` on a fresh device. Use a separate
result snapshot and report output directory when rebuilding reports locally so
that ongoing OMEN results are not overwritten.

## Verified results

- Initial full discovery ran 46 tests: 45 passed, one raised an environment error, and no
  assertions failed. The failing test is
  `test_cache_reuses_identical_attempts_and_keeps_their_theoretical_cost` in
  `tests/test_adaptive_search.py`; its `patch("torch.cuda...")` calls import
  `torch`, which this analysis environment intentionally does not include.
  The test was retained unchanged. Full authoritative regression remains an OMEN
  check in the existing environment with PyTorch.
- A fresh process imported `summarize_ts_search_v05` and confirmed that neither
  `torch` nor `mace` was present in `sys.modules`; the command-line help also loaded.
- `uv pip check` reported all 17 installed packages compatible; `git diff --check`
  passed.
- The terminal-result snapshot captured at `2026-11-23T05:42:21.763513+00:00` was
  transferred successfully and extracted into this Mac checkout. It contains 73
  completed strategy units. Running
  `TMPDIR=/tmp/.agents .venv/bin/python summarize_ts_search_v05.py --allow-incomplete`
  rebuilt the report and all four PNG/PDF figure pairs without PyTorch or MACE.
  `strategy_metrics.csv`, `summary.json` and the snapshot manifest agreed on 96
  planned / 73 completed / 23 pending units, with 67 referenced unique attempts.
  Gaussian evaluation remained incomplete and was reported that way.
- `reaction_outcomes.png` was visually inspected: all 12 holdout reactions remain
  shown for each model, attempt counts are readable, and missing fine-tuned units
  remain gray rather than disappearing from the denominator. This is an offline
  intermediate report, not proof of completion of the OMEN experiment.
- ASE emitted NumPy 2.5 array-shape deprecation warnings during existing tests.
  They did not cause the 45 successful tests to fail.

## Early DFT scheduling validation

The DFT entrypoint gained `--ready-only` so OMEN can start an already preselected
reaction once its three search strategy results are completed. Candidate coordinates,
DFT configuration, deduplication keys and final job folders use the existing logic.
Partial preparation calls `candidates(..., save_selection=False)` and writes only
`results/coordination/dft-ready-selection.json`, marked `provisional=true`. It does
not create or overwrite the canonical nine-unit `dft/selection.json`. Once all three
preselected reactions are ready, the normal complete selection is written. Existing
failed outcomes still require explicit `--retry-failed` to run again.

Two additional regressions passed: partial preparation preserves any existing
canonical selection byte for byte and produces the same keys/folders as the later
complete selection; readiness requires all three completed strategy results and
the final selection retains all nine `no_candidate` units when appropriate.
The latest full Mac run therefore ran 48 tests: 47 passed with the same single
missing-PyTorch environment error described above.

On the real snapshot, the following command prepared only `rxn0666` and did not
start Gaussian or create the canonical selection:

```bash
TMPDIR=/tmp/.agents .venv/bin/python validate_ts_dft_v05.py --ready-only --prepare-only
```

It produced the unchanged unique candidate key
`e211b73b3bcc5d621a44c0b683254b4a3393aa5d168c80b16d8bfa72c71eab79`, with job folder
`results/v0.5.0/dft/rxn0666/e211b73b3bcc5d62`. Actual early Gaussian execution is
owned by OMEN; this Mac preparation did not calculate a DFT result.

## Handoff note

The user stated that clearly specified, time-insensitive large calculation batches
may be assigned to group members, with results returned about a day later. Future
work can package such batches with fixed inputs, commands, acceptance criteria and
return files. This does not expand the current frozen experiment; no external
messages were sent as part of this reporting task.

## Complete search snapshot

The 2026-11-23T06:28:57Z terminal snapshot was reconstructed on Mac: 96/96 units, 83 unique attempts. Independent audit passed 14,084 checks with 0 issues/skips (results/coordination/mac-final-search-audit.json). Holdout original saddle+IRC 3/3/3 and strict 3/3/3; fine-tuned 5/6/6 and strict 5/5/5, all denominators 12. Strategy and reaction-matrix plots were visually checked. The shared stage-axis label was clarified without changing counts. DFT remains live on OMEN; this snapshot is not final DFT acceptance.
