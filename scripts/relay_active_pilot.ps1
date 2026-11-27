param([ValidateSet('Status','Dispatch')][string]$Action)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'worker_transport.ps1')
$cfg=Get-Content -Raw (Join-Path (Split-Path -Parent $PSScriptRoot) 'results/coordination/active-pilot-private.json')|ConvertFrom-Json
$root=$cfg.worker_root
$queue="$root/results/active-learning-pilot-20261127"
$previous="$root/results/next-compute-20261127/state.json"
$opt=@('-o','BatchMode=yes','-o','ConnectTimeout=10','-o',('HostName='+$cfg.worker_host_name),'-o',('HostKeyAlias='+$cfg.worker_host_key_alias))
function Remote([string]$code){
 $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes('$ProgressPreference="SilentlyContinue";'+$code))
 Invoke-WorkerCommand -Executable (Get-Command ssh -CommandType Application).Source -Arguments (@('-n','-T')+$opt+@($cfg.worker_ssh,'powershell','-NoProfile','-EncodedCommand',$encoded)) -TimeoutMilliseconds 20000
}
$statusCode=@'
$previous=Get-Content -Raw '__WORKER_ROOT__/results/next-compute-20261127/state.json'|ConvertFrom-Json
$active=@(Get-CimInstance Win32_Process|Where-Object {$_.Name -match '^(g16|l[0-9]+)\.exe$'})
$owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+$previous.owner_pid)
@{previous=$previous.status;owner_alive=[bool]$owner;gaussian_count=$active.Count;exists=Test-Path '__WORKER_ROOT__/results/active-learning-pilot-20261127'}|ConvertTo-Json -Compress
'@
$status=Remote ($statusCode.Replace('__WORKER_ROOT__',$root))|ConvertFrom-Json
if($Action -eq 'Status'){$status|ConvertTo-Json -Compress;return}
if($status.previous -ne 'chain_terminal' -or $status.owner_alive -or $status.gaussian_count -ne 0 -or $status.exists){throw 'Prior job not drained or active pilot already exists'}
$archive=Join-Path (Split-Path -Parent $PSScriptRoot) 'results\coordination\equal-budget-20261127\active-learning-pilot.zip'
$remoteZip="$root/results/active-learning-pilot-20261127.zip"
Invoke-WorkerCommand -Executable (Get-Command scp -CommandType Application).Source -Arguments (@('-q')+$opt+@($archive,($cfg.worker_ssh+":"+$remoteZip))) -TimeoutMilliseconds 60000|Out-Null
$launch=@'
$ErrorActionPreference='Stop'
$root='__WORKER_ROOT__'
$queue="$root/results/active-learning-pilot-20261127"
$previous=Get-Content -Raw "$root/results/next-compute-20261127/state.json"|ConvertFrom-Json
if($previous.status -ne 'chain_terminal' -or (Get-CimInstance Win32_Process -Filter ('ProcessId='+$previous.owner_pid)) -or @(Get-CimInstance Win32_Process|Where-Object {$_.Name -match '^(g16|l[0-9]+)\.exe$'}).Count){throw 'Previous owner has not drained'}
if(Test-Path $queue){throw 'Existing pilot directory'}
Expand-Archive -LiteralPath "$root/results/active-learning-pilot-20261127.zip" -DestinationPath $queue
$task=Get-Content -Raw "$queue/queue.json"|ConvertFrom-Json
if([DateTimeOffset]::Parse($task.admission_deadline_utc).UtcDateTime -le [DateTime]::UtcNow){throw 'Admission deadline elapsed'}
& "$root/scripts/run_gaussian_queue.ps1" -QueueDirectory $queue -GaussianDirectory '__GAUSSIAN_ROOT__'
'@
Remote ($launch.Replace('__WORKER_ROOT__',$root).Replace('__GAUSSIAN_ROOT__',$cfg.gaussian_directory))
