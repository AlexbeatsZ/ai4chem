# ROG compute readiness

- Date verified: 2026-11-22 (user timezone America/Los_Angeles).
- Host: Meta-ROGAlly only. These installation facts do not describe OMEN or Mac.
- Assignment: independent saved-artifact verification for the frozen ai4chem v0.5.0 experiment. No potential or DFT computations are assigned here.
- Project: `C:\Users\Meta\Project\Workspaces\ai4chem`, cloned from `https://github.com/AlexbeatsZ/ai4chem.git`; initial checkout `main` at `761e580`.
- CPU: AMD Ryzen Z1 Extreme, 8 cores / 16 logical processors. Observed CPU load was approximately 5%; Windows memory was approximately 27.7 GiB total and 6.2 GiB free at inspection. WSL Ubuntu-24.04 reported approximately 8.8 GiB available. These are transient samples.
- Gaussian and ORCA were not found in PATH, installed-application registry entries or the inspected standard Windows/WSL paths. Do not infer they are available on ROG from OMEN documentation.
- Existing tools: Windows uv and Miniforge/Mamba, Ubuntu-24.04 with system Python 3.12.3 and uv. Miniforge base Python 3.14.9 has no scientific project dependencies; it was left unchanged.
- Isolated audit environment: project `.venv`, Python 3.12.13, numpy 2.5.4 and ASE 3.29.0. ASE dependencies were resolved locally in this environment. Torch, MACE, Sella and Gaussian were not installed.
- Communication fallback: direct Tailnet SSH and uploaded ASCII PowerShell scripts, because the initial MCP run-command transport failed. No proxy configuration was changed.

The project root `AGENTS.md` and `docs/design/v0.5.0-protocol.md` were read before implementation. The experiment protocol, core computation modules, scientific outputs and frozen configuration remain unchanged. ROG receives terminal snapshots and writes a separate audit JSON; it must never resume a GPU search from copied files.
