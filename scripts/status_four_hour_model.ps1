$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$project=Split-Path -Parent $PSScriptRoot
$path=Join-Path $project 'results/four-hours-20261127/omen/state.json'
Get-Content -LiteralPath $path -Raw
