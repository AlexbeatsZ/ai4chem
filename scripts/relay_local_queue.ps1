param([Parameter(Mandatory=$true)][ValidateSet('Status','Export')][string]$Action,
    [Parameter(Mandatory=$true)][string]$WorkerRoot,[Parameter(Mandatory=$true)][string]$QueueDirectory,
    [string]$WorkerSsh='ai4chem-worker',[string]$WorkerHostName='',[string]$WorkerHostKeyAlias='')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
. (Join-Path $PSScriptRoot 'worker_transport.ps1')
foreach($value in @($WorkerRoot,$QueueDirectory,$WorkerSsh)){if($value -notmatch '^[A-Za-z0-9_./:\@-]+$'){throw 'Unsupported worker path or identity.'}}
foreach($value in @($WorkerHostName,$WorkerHostKeyAlias)){if($value -and $value -notmatch '^[A-Za-z0-9_.-]+$'){throw 'Unsupported per-command worker host override.'}}
$script=if($Action -eq 'Status'){'status_gaussian_queue.ps1'}else{'export_gaussian_queue.ps1'}
$transport=@('-o','BatchMode=yes','-o','ConnectTimeout=10')
if($WorkerHostName){$transport+=@('-o',('HostName='+$WorkerHostName))}
if($WorkerHostKeyAlias){$transport+=@('-o',('HostKeyAlias='+$WorkerHostKeyAlias))}
$arguments=@('-n','-T')+$transport+@($WorkerSsh,'powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',($WorkerRoot+'/scripts/'+$script),'-QueueDirectory',$QueueDirectory)
if($Action -eq 'Status'){
    Invoke-WorkerCommand -Executable (Get-Command ssh -CommandType Application).Source -Arguments $arguments
    return
}
$raw=Invoke-WorkerCommand -Executable (Get-Command ssh -CommandType Application).Source -Arguments $arguments
$receipt=$raw | ConvertFrom-Json
$project=Split-Path -Parent $PSScriptRoot
$queueName=Split-Path -Leaf $QueueDirectory.TrimEnd('/','\')
$output=Join-Path $project ('results\coordination\queue-returns\'+$queueName)
New-Item -ItemType Directory -Path $output -Force | Out-Null
$archive=Join-Path $output 'return.zip'
if(-not(Test-Path -LiteralPath $archive)){
    $partial=Join-Path $output 'return.partial.zip'
    Invoke-WorkerCommand -Executable (Get-Command scp -CommandType Application).Source -Arguments (@('-q')+$transport+@(($WorkerSsh+':'+$receipt.archive.Replace('\','/')),$partial)) | Out-Null
    Move-Item -LiteralPath $partial -Destination $archive
}
if((Get-Item -LiteralPath $archive).Length -ne $receipt.bytes){throw 'Terminal queue relay size differs.'}
[ordered]@{archive=$archive;bytes=$receipt.bytes;files=$receipt.files;status=$receipt.status}|ConvertTo-Json
