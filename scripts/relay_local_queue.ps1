param([Parameter(Mandatory=$true)][ValidateSet('Status','Export')][string]$Action,
    [Parameter(Mandatory=$true)][string]$WorkerRoot,[Parameter(Mandatory=$true)][string]$QueueDirectory,
    [string]$WorkerSsh='ai4chem-worker')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
foreach($value in @($WorkerRoot,$QueueDirectory,$WorkerSsh)){if($value -notmatch '^[A-Za-z0-9_./:\@-]+$'){throw 'Unsupported worker path or identity.'}}
$script=if($Action -eq 'Status'){'status_gaussian_queue.ps1'}else{'export_gaussian_queue.ps1'}
$raw=& ssh -n -T -o BatchMode=yes -o ConnectTimeout=10 $WorkerSsh powershell -NoProfile -ExecutionPolicy Bypass -File ($WorkerRoot+'/scripts/'+$script) -QueueDirectory $QueueDirectory
if($LASTEXITCODE -ne 0){throw 'Queue observation/export failed.'}
if($Action -eq 'Status'){$raw;return}
$receipt=($raw -join "`n") | ConvertFrom-Json
$project=Split-Path -Parent $PSScriptRoot
$output=Join-Path $project 'results\overnight-20261124\roommate-return'
New-Item -ItemType Directory -Path $output -Force | Out-Null
$archive=Join-Path $output 'return.zip'
if(-not(Test-Path -LiteralPath $archive)){
    $partial=Join-Path $output 'return.partial.zip'
    & scp -q -o BatchMode=yes -o ConnectTimeout=10 ($WorkerSsh+':'+$receipt.archive.Replace('\','/')) $partial
    if($LASTEXITCODE -ne 0){throw 'Terminal queue relay failed.'}
    Move-Item -LiteralPath $partial -Destination $archive
}
if((Get-Item -LiteralPath $archive).Length -ne $receipt.bytes){throw 'Terminal queue relay size differs.'}
[ordered]@{archive=$archive;bytes=$receipt.bytes;files=$receipt.files;status=$receipt.status}|ConvertTo-Json
