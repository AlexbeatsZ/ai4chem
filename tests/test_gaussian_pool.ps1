param([Parameter(Mandatory=$true)][string]$TemporaryRoot)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
. (Join-Path $project 'scripts\run_gaussian_pool.ps1') -FunctionsOnly
$config=Read-Json (Join-Path $project 'configs\roommate-pool-20261124.json')
Assert-PoolSettings $config
$now=[DateTime]::UtcNow;$deadline=$now.AddHours(24)
if(-not(Can-Admit $config 1 8 $deadline $now $false)){throw 'Second worker was refused despite capacity.'}
if(Can-Admit $config 2 8 $deadline $now $false){throw 'Third worker was admitted.'}
if(Can-Admit $config 1 5.9 $deadline $now $false){throw 'Low-memory worker was admitted.'}
if(Can-Admit $config 1 8 $now $now $false){throw 'Expired wall budget admitted a worker.'}
if(Can-Admit $config 1 8 $deadline $now $true){throw 'Failure block admitted a worker.'}
$folder='C:\fixture\chain'
$good=[pscustomobject]@{Name='powershell.exe';CommandLine='powershell -File C:\scripts\run_gaussian_chain.ps1 -JobDirectory C:\fixture\chain -Worker'}
Assert-ChildIdentity $good 'run_gaussian_chain.ps1' $folder
foreach($bad in @(
    [pscustomobject]@{Name='powershell.exe';CommandLine='powershell -File C:\scripts\run_gaussian_chain.ps1 -JobDirectory C:\other\chain -Worker'},
    [pscustomobject]@{Name='powershell.exe';CommandLine='powershell -File C:\scripts\run_gaussian_chain.ps1 -JobDirectory C:\fixture\chain'},
    [pscustomobject]@{Name='other.exe';CommandLine=$good.CommandLine}
)){
    $rejected=$false;try{Assert-ChildIdentity $bad 'run_gaussian_chain.ps1' $folder}catch{$rejected=$true}
    if(-not $rejected){throw 'Unrelated child identity was adopted.'}
}
$claims=Join-Path $TemporaryRoot ('claims-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $claims|Out-Null
Claim-Item $claims 'rxn0255'
$first=[IO.File]::ReadAllText((Join-Path $claims 'rxn0255.json'))
$rejected=$false;try{Claim-Item $claims 'rxn0255'}catch{$rejected=$true}
if(-not $rejected -or [IO.File]::ReadAllText((Join-Path $claims 'rxn0255.json')) -cne $first){throw 'Duplicate claim overwrote identity.'}
foreach($pair in @(@('max_workers',3),@('minimum_available_GiB_before_launch',4),@('wall_hours_from_pool_start',100))){
    $modified=Read-Json (Join-Path $project 'configs\roommate-pool-20261124.json');$modified.($pair[0])=$pair[1]
    $rejected=$false;try{Assert-PoolSettings $modified}catch{$rejected=$true}
    if(-not $rejected){throw 'Unsafe resource amendment was accepted.'}
}
[ordered]@{status='PASS';second_worker_allowed=$true;third_worker_refused=$true;memory_gate=$true;deadline_gate=$true;
    failure_gate=$true;unrelated_child_refused=$true;duplicate_claim_preserved=$true;unsafe_config_refused=$true}|ConvertTo-Json
