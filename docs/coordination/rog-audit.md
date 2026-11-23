# Independent artifact audit and ROG handoff

- Last verified: 2026-11-22, America/Los_Angeles.
- Scope: saved artifacts from the frozen v0.5.0 experiment; no model, Gaussian or other scientific computation was started on ROG.
- ROG project: `C:\Users\Meta\Project\Workspaces\ai4chem`.
- Isolated Python: `.venv\Scripts\python.exe` (Python 3.12.13, numpy 2.5.4, ASE 3.29.0).
- Auditor: `scripts\audit_v05_artifacts.py`. Initial upload and AST validation on ROG succeeded. The final small cache-reference check was subsequently incorporated in the Mac copy; it has not been confirmed uploaded to ROG after the connection loss.

## Audit implementation

The auditor imports only the Python standard library, numpy and ASE. It does not import the project search, policy, model or reporting modules. Its required arguments are `--root` and `--output`. Source artifacts are read-only, previous source JSON files cannot be overwritten as the requested output, and existing saved paths are relocated using their `results/v0.5.0/` suffix.

It checks the frozen manifest and policy, input and saved atom mapping, finite coordinates/energies/forces, attempt identities, frozen settings, model/software provenance recorded with the attempts, endpoint identity, initial/final bands, perturbation magnitude and translation removal, unchanged perturbation endpoints, continuation parent geometry, Hessian symmetry, saved and independently reprojected frequencies, declared step budgets, strategy selection, shared endpoint time and cumulative strategy costs. Model weights are not read. SHA-256 is used only for declared cache identities and continuation-band identity, not as a separate transfer audit.

Only terminal attempts referenced by completed strategy records are inspected. Missing strategy units are pending and remain outside this partial integrity result. An integrity pass is not a claim that a reaction succeeded or that the matrix is complete.

## Verified execution on Mac

The same independent auditor ran on the terminal snapshot already downloaded by the coordinator to the Mac checkout. This was a Mac fallback, not a ROG execution result.

- Completed strategy units audited: 73 of 96 planned (24 development, 49 holdout).
- Unique referenced terminal attempts: 67.
- Input reactions checked: 16.
- Artifact checks: 11,349; issues: 0; skipped checks: 0.
- `integrity_passed=true`, `matrix_complete=false`.
- The parent coordinator independently ran the integrated script against the same snapshot and observed the same counts and exit code 0.

## ROG connection and incomplete transfer

The ROG-side download command was started with the completed OMEN snapshot at `C:\Users\Meta\AppData\Local\Temp\.agents\ai4chem-v05-snapshot.zip`, whose coordinator-reported exported payload contained 1,654 files and 73 terminal strategy units. The command was to copy it to the same approved ROG Temp path, extract into the ROG project, then run the auditor. The SSH connection stopped responding before any download/extraction/audit completion output was observed. Successful transfer, extraction and ROG execution have not been established.

Subsequent direct SSH probes to both `100.106.169.46:22` and the known LAN address `192.168.31.32:22` timed out after ten seconds. Even a small readiness-log transfer was left unconfirmed. The task-owned stalled Mac SSH wrapper was terminated after the bounded probes failed; native ROG child-process termination could not be confirmed offline. Do not interpret the absence of a ROG audit JSON as an artifact failure.

On reconnect, first inspect whether the snapshot zip and `docs\coordination\rog-artifact-audit.json` exist, and inspect task-owned process command lines for `rog-ai4chem-snapshot.ps1` or its `scp` child before launching another transfer. If the zip is partial, replace it with a fresh terminal snapshot rather than extracting it. Copy the latest coordinator-owned auditor to ROG before the next run. Do not resume GPU searches or modify scientific records from the ROG snapshot.

Task-owned ROG temporary files to clean up after reconnect: `rog-ai4chem-inspect.ps1`, `rog-ai4chem-env.ps1`, `rog-ai4chem-write-audit.ps1`, `rog-ai4chem-snapshot.ps1`, `audit_v05_artifacts.py`, and the snapshot zip, all beneath `C:\Users\Meta\AppData\Local\Temp\.agents\`. The project `.venv`, auditor and coordination notes are intentional persistent project files.
