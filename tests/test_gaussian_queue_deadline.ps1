param([Parameter(Mandatory=$true)][string]$TemporaryRoot)
$ErrorActionPreference='Stop'
$fixture=Join-Path $TemporaryRoot ('queue-deadline-'+[Guid]::NewGuid().ToString('N'))
$scripts=Join-Path $fixture 'scripts';New-Item -ItemType Directory -Path $scripts -Force|Out-Null
Copy-Item -LiteralPath (Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts\run_gaussian_queue.ps1') -Destination (Join-Path $scripts 'run_gaussian_queue.ps1')
[IO.File]::WriteAllText((Join-Path $fixture 'run_gaussian_job.ps1'),@'
param([string]$InputFile,[string]$GaussianDirectory)
Start-Sleep -Seconds 6
[IO.File]::WriteAllText((Join-Path (Split-Path -Parent $InputFile) 'force.log'),'SCF Done: synthetic; Forces (Hartrees/Bohr)')
'@)
$previous=Join-Path $fixture 'previous';New-Item -ItemType Directory -Path $previous|Out-Null
[IO.File]::WriteAllText((Join-Path $previous 'queue-state.json'),'"unused"')
@{owner_pid=99999999;status='prepared_pool_exhausted'}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $previous 'queue-state.json')
$powershell=Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
foreach($case in @('declined','expired','drain')){
    $queue=Join-Path $fixture $case;New-Item -ItemType Directory -Path $queue|Out-Null
    $gate=Join-Path $queue 'gate.json'
    @{protocol='three-host-until-0800-20261126';allow_label_extension=($case -ne 'declined')}|ConvertTo-Json|Set-Content -LiteralPath $gate
    [IO.File]::WriteAllText(($gate+'.ready'),'ready')
    $items=@();foreach($i in 1..3){$name='label'+$i;[IO.File]::WriteAllText((Join-Path $queue ($name+'.gjf')),'synthetic');$items+=@{name=$name;reaction='fixture';input=($name+'.gjf')}}
    $deadline=if($case -eq 'expired'){[DateTime]::UtcNow.AddSeconds(-1)}elseif($case -eq 'drain'){[DateTime]::UtcNow.AddSeconds(5)}else{[DateTime]::UtcNow.AddSeconds(60)}
    @{chains=@();force_items=$items;target_compute_hours=12;predecessor_queue=$previous;gate_file=$gate;admission_deadline_utc=$deadline.ToString('o')}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $queue 'queue.json')
    $process=Start-Process -FilePath $powershell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $scripts 'run_gaussian_queue.ps1'),'-QueueDirectory',$queue,'-Worker') -WindowStyle Hidden -PassThru
    if(-not $process.WaitForExit(20000)){Stop-Process -Id $process.Id;throw 'Synthetic finite queue did not return.'}
    $state=[IO.File]::ReadAllText((Join-Path $queue 'queue-state.json'))|ConvertFrom-Json
    if($case -eq 'declined' -and ($state.status -ne 'pilot_gate_declined' -or $state.outcomes.Count)){throw 'Declined pilot admitted chemistry.'}
    if($case -eq 'expired' -and ($state.status -ne 'admission_deadline_reached' -or $state.outcomes.Count)){throw 'Expired queue admitted chemistry.'}
    if($case -eq 'drain' -and ($state.status -ne 'admission_deadline_reached' -or $state.outcomes.Count -ne 1)){throw 'Deadline failed to drain exactly one synthetic job.'}
}
[ordered]@{status='PASS';declined_gate_admits_nothing=$true;expired_gate_admits_nothing=$true;deadline_drains_without_extra_admission=$true;synthetic_only=$true}|ConvertTo-Json
