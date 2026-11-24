param([Parameter(Mandatory=$true)][string]$QueueDirectory)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$QueueDirectory=(Resolve-Path -LiteralPath $QueueDirectory).Path
$state=[IO.File]::ReadAllText((Join-Path $QueueDirectory 'queue-state.json')) | ConvertFrom-Json
$processes=@(Get-CimInstance Win32_Process)
$active=@($processes | Where-Object {$_.Name -match '^(g16|l[0-9]+)\.exe$'} | Select-Object ProcessId,Name)
[ordered]@{host=$state.host;owner_pid=$state.owner_pid;owner_alive=[bool](@($processes | Where-Object {$_.ProcessId -eq $state.owner_pid}).Count);
    status=$state.status;stage=$state.stage;compute_seconds=$state.compute_seconds;updated_at_utc=$state.updated_at_utc;
    completed_items=$state.outcomes.Count;last_outcomes=@($state.outcomes | Select-Object -Last 3);error=$state.error;
    gaussian_processes=$active} | ConvertTo-Json -Depth 10
