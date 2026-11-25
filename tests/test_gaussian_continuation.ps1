param([Parameter(Mandatory=$true)][string]$TemporaryRoot)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$fixture=Join-Path $TemporaryRoot ('continuation-'+[Guid]::NewGuid().ToString('N'))
foreach($name in @('scripts','previous','queue\inputs','queue\claims','queue\forces\old')){
    New-Item -ItemType Directory -Path (Join-Path $fixture $name) -Force|Out-Null
}
$queue=Join-Path $fixture 'queue'
Copy-Item -LiteralPath (Join-Path $project 'scripts\run_gaussian_pool.ps1') -Destination (Join-Path $fixture 'scripts\run_gaussian_pool.ps1')
foreach($name in @('status_gaussian_queue.ps1','export_gaussian_queue.ps1')){
    Copy-Item -LiteralPath (Join-Path $project ('scripts\'+$name)) -Destination (Join-Path $fixture ('scripts\'+$name))
}
Copy-Item -LiteralPath (Join-Path $project 'configs\roommate-pool-20261124.json') -Destination (Join-Path $fixture 'execution.json')
$execution=[IO.File]::ReadAllText((Join-Path $fixture 'execution.json'))|ConvertFrom-Json
$execution.poll_seconds=10
[IO.File]::WriteAllText((Join-Path $fixture 'execution.json'),($execution|ConvertTo-Json))
$previous=@'
param([string]$QueueDirectory,[switch]$Worker)
$lock=[IO.File]::Open((Join-Path $QueueDirectory 'queue.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$state=@{owner_pid=$PID;status='running';active=@();outcomes=@(@{name='old';status='failed';kind='force';elapsed_seconds=2});compute_seconds=2;started_at_utc=[DateTime]::UtcNow.ToString('o')}
[IO.File]::WriteAllText((Join-Path $QueueDirectory 'queue-state.json'),($state|ConvertTo-Json -Depth 8))
Start-Sleep -Seconds 6
$state.status='wall_budget_reached'
[IO.File]::WriteAllText((Join-Path $QueueDirectory 'queue-state.json'),($state|ConvertTo-Json -Depth 8))
$lock.Dispose()
'@
[IO.File]::WriteAllText((Join-Path $fixture 'previous\run_gaussian_pool.ps1'),$previous)
$force=@'
param([string]$InputFile,[string]$GaussianDirectory)
[IO.File]::WriteAllText((Join-Path (Split-Path -Parent $InputFile) 'force.log'),'SCF Done: synthetic; Forces (Hartrees/Bohr)')
'@
[IO.File]::WriteAllText((Join-Path $fixture 'run_gaussian_job.ps1'),$force)
[IO.File]::WriteAllText((Join-Path $queue 'inputs\force.gjf'),'SYNTHETIC')
[IO.File]::WriteAllText((Join-Path $queue 'claims\old.json'),'ORIGINAL CLAIM')
$oldReceipt='{"name":"old","status":"failed"}'
[IO.File]::WriteAllText((Join-Path $queue 'forces\old\receipt.json'),$oldReceipt)
$task=@{chains=@();force_items=@(@{name='old';reaction='fixture';input='inputs/force.gjf'},@{name='fresh1';reaction='fixture';input='inputs/force.gjf'},@{name='fresh2';reaction='fixture';input='inputs/force.gjf'})}
[IO.File]::WriteAllText((Join-Path $queue 'queue.json'),($task|ConvertTo-Json -Depth 8))
$continuation=Join-Path $fixture 'continuation.json'
[IO.File]::WriteAllText($continuation,(@{admission_deadline_utc=[DateTime]::UtcNow.AddSeconds(90).ToString('o')}|ConvertTo-Json))
$powershell=Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$old=Start-Process -FilePath $powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $fixture 'previous\run_gaussian_pool.ps1')`" -QueueDirectory `"$queue`" -Worker" -WindowStyle Hidden -PassThru
$owner=$null
try{
    $deadline=[DateTime]::UtcNow.AddSeconds(10)
    while(-not(Test-Path -LiteralPath (Join-Path $queue 'queue-state.json'))){if([DateTime]::UtcNow -gt $deadline){throw 'Synthetic predecessor did not start.'};Start-Sleep -Milliseconds 200}
    & (Join-Path $fixture 'scripts\run_gaussian_pool.ps1') -QueueDirectory $queue -ExecutionFile (Join-Path $fixture 'execution.json') -ContinuationFile $continuation -ContinueAfterOwner $old.Id | Out-Null
    $start=[IO.File]::ReadAllText((Join-Path $queue 'continuation-start.json'))|ConvertFrom-Json
    $owner=$start.owner_pid
    if(-not(Get-CimInstance Win32_Process -Filter ('ProcessId='+$old.Id))){throw 'Original predecessor was interrupted.'}
    $deadline=[DateTime]::UtcNow.AddSeconds(55)
    $handoffChecked=$false
    while(Get-CimInstance Win32_Process -Filter ('ProcessId='+$owner)){
        if([DateTime]::UtcNow -ge $deadline){throw 'Continuation failed to terminate within55s.'}
        $current=[IO.File]::ReadAllText((Join-Path $queue 'queue-state.json'))|ConvertFrom-Json
        if(-not $handoffChecked -and $current.status -eq 'wall_budget_reached'){
            $observed=(& (Join-Path $fixture 'scripts\status_gaussian_queue.ps1') -QueueDirectory $queue)|ConvertFrom-Json
            if(-not $observed.owner_alive -or $observed.continuation_owner_pid -ne $owner){throw 'Handoff was reported as owner loss.'}
            $refused=$false
            try{& (Join-Path $fixture 'scripts\export_gaussian_queue.ps1') -QueueDirectory $queue|Out-Null}catch{$refused=$true}
            if(-not $refused){throw 'Handoff exported incomplete queue evidence.'}
            $handoffChecked=$true
        }
        Start-Sleep -Seconds 1
    }
    $state=[IO.File]::ReadAllText((Join-Path $queue 'queue-state.json'))|ConvertFrom-Json
    if($state.status -ne 'prepared_pool_exhausted' -or $state.outcomes.Count -ne 3){throw ('Continuation failed: '+($state|ConvertTo-Json -Depth 8))}
    if([IO.File]::ReadAllText((Join-Path $queue 'forces\old\receipt.json')) -cne $oldReceipt){throw 'Failed evidence was overwritten.'}
    if([IO.File]::ReadAllText((Join-Path $queue 'claims\old.json')) -cne 'ORIGINAL CLAIM'){throw 'Old claim was replaced.'}
    if(@(Get-ChildItem -LiteralPath (Join-Path $queue 'claims') -File).Count -ne 3){throw 'Claims differ from three exact items.'}
    if(-not(Test-Path -LiteralPath (Join-Path $queue 'pre-continuation-state.json'))){throw 'Original terminal state was not retained.'}
    $marker=[IO.File]::ReadAllText((Join-Path $queue 'continuation-state.json'))|ConvertFrom-Json
    if($marker.status -ne 'prepared_pool_exhausted'){throw 'Continuation marker is not terminal.'}
    if(-not $handoffChecked){throw 'Synthetic handoff observation was not exercised.'}
    [ordered]@{status='PASS';predecessor_drained_naturally=$true;failed_claim_preserved=$true;new_items_once=$true;
        original_state_retained=$true;no_early_export=$handoffChecked;absolute_deadline=$state.admission_deadline_utc;synthetic_only=$true}|ConvertTo-Json
}finally{
    foreach($processId in @($old.Id,$owner)){
        if($processId -and (Get-CimInstance Win32_Process -Filter ('ProcessId='+$processId))){Stop-Process -Id $processId -ErrorAction SilentlyContinue}
    }
}
