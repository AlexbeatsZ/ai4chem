param([Parameter(Mandatory=$true)][string]$QueueDirectory)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$QueueDirectory=(Resolve-Path -LiteralPath $QueueDirectory).Path
$state=[IO.File]::ReadAllText((Join-Path $QueueDirectory 'queue-state.json')) | ConvertFrom-Json
$processes=@(Get-CimInstance Win32_Process)
$active=@($processes | Where-Object {$_.Name -match '^(g16|l[0-9]+)\.exe$'} | Select-Object ProcessId,Name)
$ownerAlive=[bool](@($processes | Where-Object {$_.ProcessId -eq $state.owner_pid}).Count)
$continuationOwner=$null
$continuationFile=Join-Path $QueueDirectory 'continuation-start.json'
$reportedDeadline=$state.admission_deadline_utc
if(Test-Path -LiteralPath $continuationFile){
    $continuation=[IO.File]::ReadAllText($continuationFile)|ConvertFrom-Json
    $continuationOwner=@($processes|Where-Object {$_.ProcessId -eq $continuation.owner_pid -and
        $_.Name -eq 'powershell.exe' -and $_.CommandLine.Contains('run_gaussian_pool.ps1') -and
        $_.CommandLine.Contains('-ContinuationFile') -and $_.CommandLine.Replace('/','\').Contains($QueueDirectory.Replace('/','\'))})
    if($continuationOwner.Count){$ownerAlive=$true;$reportedDeadline=$continuation.deadline_utc}
}
$handoff=Join-Path $QueueDirectory 'pool-start.json'
if(-not $ownerAlive -and (Test-Path -LiteralPath $handoff)){
    $transition=[IO.File]::ReadAllText($handoff)|ConvertFrom-Json
    if($transition.new_owner_pid -and @($processes|Where-Object {$_.ProcessId -eq $transition.new_owner_pid}).Count){
        $ownerAlive=$true
    }
}
[ordered]@{host=$state.host;owner_pid=$state.owner_pid;owner_alive=$ownerAlive;
    status=$state.status;stage=$state.stage;compute_seconds=$state.compute_seconds;updated_at_utc=$state.updated_at_utc;
    completed_items=$state.outcomes.Count;last_outcomes=@($state.outcomes | Select-Object -Last 3);error=$state.error;
    gaussian_processes=$active;pool_active=$state.active;pending_items=$state.pending_items;admission_deadline_utc=$reportedDeadline;
    current_phase_deadline_utc=$state.admission_deadline_utc;continuation_owner_pid=if($continuationOwner.Count){$continuationOwner[0].ProcessId}else{$null};
    handoff_owner_alive=$ownerAlive} | ConvertTo-Json -Depth 10
