param([Parameter(Mandatory=$true)][string]$TemporaryRoot)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$fixture=Join-Path $TemporaryRoot ('gaussian-queue-fixture-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path (Join-Path $fixture 'scripts') -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $project 'scripts\run_gaussian_queue.ps1') -Destination (Join-Path $fixture 'scripts\run_gaussian_queue.ps1')
$fake=@'
param([string]$InputFile,[string]$GaussianDirectory)
if([IO.File]::ReadAllText($InputFile).Contains('FAIL')){throw 'Synthetic Gaussian failure'}
[IO.File]::WriteAllText((Join-Path (Split-Path -Parent $InputFile) 'force.log'),'SCF Done: synthetic fixture; Forces (Hartrees/Bohr)')
'@
[IO.File]::WriteAllText((Join-Path $fixture 'run_gaussian_job.ps1'),$fake)
$runner=Join-Path $fixture 'scripts\run_gaussian_queue.ps1'
foreach($mode in @('success','failure')) {
    $directory=Join-Path $fixture $mode
    New-Item -ItemType Directory -Path (Join-Path $directory 'inputs') -Force | Out-Null
    $items=@()
    foreach($n in 1..4){
        $name="job$n";$items+=@{name=$name;input="inputs/$name.gjf";reaction='synthetic'}
        $text=if($mode -eq 'failure'){'FAIL'}else{'SUCCESS'}
        [IO.File]::WriteAllText((Join-Path $directory "inputs/$name.gjf"),$text)
    }
    $hours=if($mode -eq 'success'){0.0000000001}else{24}
    $task=@{chains=@();force_items=$items;target_compute_hours=$hours}
    [IO.File]::WriteAllText((Join-Path $directory 'queue.json'),($task|ConvertTo-Json -Depth 8))
    & $runner -QueueDirectory $directory -Worker
    $state=[IO.File]::ReadAllText((Join-Path $directory 'queue-state.json')) | ConvertFrom-Json
    if($mode -eq 'success' -and ($state.status -ne 'compute_budget_reached' -or $state.outcomes.Count -ne 1)){throw 'Fixed compute horizon did not stop after first completed fixture.'}
    if($mode -eq 'failure' -and ($state.status -ne 'blocked' -or $state.outcomes.Count -ne 3)){throw 'Three failures did not block; retry or later job was launched.'}
    $rejected=$false
    try{& $runner -QueueDirectory $directory | Out-Null}catch{$rejected=$_.Exception.Message -like '*Existing queue evidence*'}
    if(-not $rejected){throw 'Repeated launcher accepted existing queue evidence.'}
    $before=[IO.File]::ReadAllText((Join-Path $directory 'queue-state.json'))
    $rejected=$false
    try{& $runner -QueueDirectory $directory -Worker | Out-Null}catch{$rejected=$_.Exception.Message -like '*Existing queue evidence*'}
    if(-not $rejected -or [IO.File]::ReadAllText((Join-Path $directory 'queue-state.json')) -ne $before){throw 'Direct worker retry changed retained evidence.'}
}
$terminal=Join-Path $fixture 'terminal'
New-Item -ItemType Directory -Path (Join-Path $terminal 'forces\label') -Force | Out-Null
[IO.File]::WriteAllText((Join-Path $terminal 'queue-state.json'),'{"status":"compute_budget_reached","owner_pid":2147483647}')
[IO.File]::WriteAllText((Join-Path $terminal 'forces\label\receipt.json'),'{"id":"one"}')
[IO.File]::WriteAllText((Join-Path $terminal 'forces\label\force.log'),'synthetic-terminal')
$exporter=Join-Path $project 'scripts\export_gaussian_queue.ps1'
$first=(& $exporter -QueueDirectory $terminal | ConvertFrom-Json)
$stamp=(Get-Item -LiteralPath $first.archive).LastWriteTimeUtc
$second=(& $exporter -QueueDirectory $terminal | ConvertFrom-Json)
if($first.files -ne 3 -or $first.bytes -ne $second.bytes -or (Get-Item -LiteralPath $first.archive).LastWriteTimeUtc -ne $stamp){throw 'Nested terminal export was not reused.'}
[IO.File]::WriteAllText((Join-Path $terminal 'forces\label\receipt.json'),'{"id":"two"}')
$rejected=$false
try{& $exporter -QueueDirectory $terminal | Out-Null}catch{$rejected=$_.Exception.Message -like '*identity differs*'}
if(-not $rejected -or (Get-Item -LiteralPath $first.archive).LastWriteTimeUtc -ne $stamp){throw 'Changed terminal identity was accepted or archive was replaced.'}
[ordered]@{status='PASS';budget_stops_admission=$true;three_failures_stop=$true;existing_output_rejected=$true;direct_worker_retry_rejected=$true;nested_export_reused=$true;changed_identity_rejected=$true;synthetic_only=$true}|ConvertTo-Json
