param([Parameter(Mandatory=$true)][string]$TemporaryRoot)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$fixture=Join-Path $TemporaryRoot ('pool-integration-'+[Guid]::NewGuid().ToString('N'))
$scripts=Join-Path $fixture 'scripts';New-Item -ItemType Directory -Path $scripts -Force|Out-Null
Copy-Item -LiteralPath (Join-Path $project 'scripts\run_gaussian_pool.ps1') -Destination (Join-Path $scripts 'run_gaussian_pool.ps1')
$fake=@'
param([string]$JobDirectory,[string]$GaussianDirectory,[switch]$Worker)
$record=@{owner_pid=$PID;status='running';started_at_utc=[DateTime]::UtcNow.ToString('o')}
[IO.File]::WriteAllText((Join-Path $JobDirectory 'worker-state.json'),($record|ConvertTo-Json))
$seconds=if($JobDirectory.EndsWith('adopted')){25}else{3}
Start-Sleep -Seconds $seconds
$record.status='raw_chain_completed';$record['elapsed_seconds']=$seconds
[IO.File]::WriteAllText((Join-Path $JobDirectory 'worker-state.json'),($record|ConvertTo-Json))
'@
[IO.File]::WriteAllText((Join-Path $scripts 'run_gaussian_chain.ps1'),$fake)
$force=@'
param([string]$InputFile,[string]$GaussianDirectory)
[IO.File]::WriteAllText((Join-Path (Split-Path -Parent $InputFile) 'force.log'),'SCF Done: synthetic; Forces (Hartrees/Bohr)')
'@
[IO.File]::WriteAllText((Join-Path $fixture 'run_gaussian_job.ps1'),$force)
$queue=Join-Path $fixture 'queue'
foreach($name in @('adopted','second','inputs')){New-Item -ItemType Directory -Path (Join-Path $queue $name) -Force|Out-Null}
foreach($name in @('adopted','second')){[IO.File]::WriteAllText((Join-Path $queue ($name+'\task.json')),'{"configuration":{"dft_nproc":4,"dft_mem":"2GB"}}')}
[IO.File]::WriteAllText((Join-Path $queue 'inputs\force.gjf'),'SYNTHETIC')
$task=@{chains=@(@{job_id='adopted';directory='adopted'},@{job_id='second';directory='second'});force_items=@(@{name='label1';reaction='fixture';input='inputs/force.gjf'})}
[IO.File]::WriteAllText((Join-Path $queue 'queue.json'),($task|ConvertTo-Json -Depth 8))
$config=[IO.File]::ReadAllText((Join-Path $project 'configs\roommate-pool-20261124.json'))|ConvertFrom-Json
$config.poll_seconds=10
$execution=Join-Path $fixture 'execution.json';[IO.File]::WriteAllText($execution,($config|ConvertTo-Json))
$powershell=Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$adoptedPath=Join-Path $queue 'adopted'
$child=Start-Process -FilePath $powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $scripts 'run_gaussian_chain.ps1')`" -JobDirectory `"$adoptedPath`" -Worker" -WindowStyle Hidden -PassThru
try{
    $deadline=[DateTime]::UtcNow.AddSeconds(10)
    while(-not(Test-Path -LiteralPath (Join-Path $adoptedPath 'worker-state.json'))){if([DateTime]::UtcNow -gt $deadline){throw 'Fixture child did not start.'};Start-Sleep -Milliseconds 200}
    $receipt=@{adopted=@(@{job_id='adopted'});previous_state=@{started_at_utc=[DateTime]::UtcNow.ToString('o');outcomes=@();compute_seconds=0}}
    [IO.File]::WriteAllText((Join-Path $queue 'pool-start.json'),($receipt|ConvertTo-Json -Depth 8))
    $log=Join-Path $fixture 'pool.log';$err=Join-Path $fixture 'pool.err'
    $pool=Start-Process -FilePath $powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $scripts 'run_gaussian_pool.ps1')`" -QueueDirectory `"$queue`" -ExecutionFile `"$execution`" -Worker" -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
    if(-not $pool.WaitForExit(60000)){Stop-Process -Id $pool.Id;throw 'Synthetic pool failed to finish within60s.'}
    $state=[IO.File]::ReadAllText((Join-Path $queue 'queue-state.json'))|ConvertFrom-Json
    if($state.status -ne 'prepared_pool_exhausted' -or $state.outcomes.Count -ne 3){throw ('Synthetic pool failed: '+($state|ConvertTo-Json -Depth 10))}
    $first=[IO.File]::ReadAllText((Join-Path $adoptedPath 'worker-state.json'))|ConvertFrom-Json
    $second=[IO.File]::ReadAllText((Join-Path $queue 'second\worker-state.json'))|ConvertFrom-Json
    if($first.owner_pid -ne $child.Id){throw 'Adopted child was replaced.'}
    if(([DateTime]$second.started_at_utc-[DateTime]$first.started_at_utc).TotalSeconds -ge 20){throw 'Second child was not concurrent with adopted child.'}
    if(@(Get-ChildItem -LiteralPath (Join-Path $queue 'claims') -File).Count -ne 3){throw 'Task claims differ from exactly three jobs.'}
    [ordered]@{status='PASS';adopted_pid_preserved=$true;two_workers_actually_overlap=$true;all_three_tasks_once=$true;queue_exhausted_cleanly=$true;synthetic_only=$true}|ConvertTo-Json
}finally{
    if(Get-CimInstance Win32_Process -Filter ('ProcessId='+$child.Id)){Stop-Process -Id $child.Id -ErrorAction SilentlyContinue}
}
