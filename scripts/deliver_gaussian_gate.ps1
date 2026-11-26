param([Parameter(Mandatory=$true)][string]$SourceFile,[Parameter(Mandatory=$true)][string]$QueueDirectory,
    [string]$WorkerHostName='GUGUGAGA.local',[string]$WorkerHostKeyAlias='192.168.31.41')
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'worker_transport.ps1')
foreach($value in @($SourceFile,$QueueDirectory,$WorkerHostName,$WorkerHostKeyAlias)){
    if($value -notmatch '^[A-Za-z0-9_./:\-]+$'){throw 'Unsupported gate transfer identity.'}
}
$gate=[IO.File]::ReadAllText($SourceFile)|ConvertFrom-Json
if($gate.protocol -ne 'three-host-until-0800-20261126' -or $null -eq $gate.allow_label_extension){throw 'Unexpected pilot gate identity.'}
$ready=$SourceFile+'.ready'
[IO.File]::WriteAllText($ready,'ready',[Text.UTF8Encoding]::new($false))
$options=@('-q','-o','BatchMode=yes','-o','ConnectTimeout=10','-o',('HostName='+$WorkerHostName),'-o',('HostKeyAlias='+$WorkerHostKeyAlias))
foreach($pair in @(@($SourceFile,'adaptation-gate.json'),@($ready,'adaptation-gate.json.ready'))){
    Invoke-WorkerCommand -Executable (Get-Command scp -CommandType Application).Source -Arguments ($options+@($pair[0],('ai4chem-worker:'+$QueueDirectory+'/'+$pair[1]))) | Out-Null
}
[ordered]@{gate_delivered=$true;allow_label_extension=$gate.allow_label_extension}|ConvertTo-Json
