param(
    [string]$QueueDirectory,[string]$ExecutionFile,[string]$GaussianDirectory='C:\G16W',
    [switch]$Worker,[int]$TakeoverOwner=0,[string]$ForceItem='', [switch]$FunctionsOnly
)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)

function Read-Json([string]$Path){return ([IO.File]::ReadAllText($Path)|ConvertFrom-Json)}
function Write-Json([string]$Path,$Value){
    $temporary=$Path+'.tmp'
    [IO.File]::WriteAllText($temporary,($Value|ConvertTo-Json -Depth 18),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}
function Assert-PoolSettings($Config){
    if($Config.max_workers -ne 2 -or $Config.threads_per_worker -ne 4 -or $Config.gaussian_memory_GB -ne 2){throw 'Unsupported resource envelope.'}
    if($Config.minimum_available_GiB_before_launch -lt 6 -or $Config.minimum_available_GiB_to_keep_admission -lt 3){throw 'Admission headroom cannot be reduced.'}
    if($Config.wall_hours_from_pool_start -lt 18 -or $Config.wall_hours_from_pool_start -gt 24){throw 'Use a finite18-24h wall horizon.'}
    if($Config.poll_seconds -lt 10 -or $Config.failure_limit -ne 3){throw 'Unsupported monitoring/failure policy.'}
}
function Can-Admit($Config,[int]$ActiveCount,[double]$AvailableGiB,[DateTime]$Deadline,[DateTime]$Now,[bool]$Blocked){
    return (-not $Blocked -and $ActiveCount -lt $Config.max_workers -and
        $AvailableGiB -ge $Config.minimum_available_GiB_before_launch -and $Now -lt $Deadline)
}
function Assert-ChildIdentity($Process,[string]$ExpectedScript,[string]$Folder){
    if(-not $Process -or $Process.Name -ne 'powershell.exe' -or
        -not $Process.CommandLine.Contains($ExpectedScript) -or
        -not $Process.CommandLine.Replace('/','\').Contains($Folder.Replace('/','\')) -or
        $Process.CommandLine -notmatch ' -Worker(?:\s|$)'){throw 'Recorded child has no matching live identity.'}
}
function Claim-Item([string]$Directory,[string]$Name){
    if($Name -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid claim identity.'}
    $path=Join-Path $Directory ($Name+'.json')
    $stream=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try{
        $bytes=[Text.Encoding]::UTF8.GetBytes((@{name=$Name;owner_pid=$PID;claimed_at_utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json))
        $stream.Write($bytes,0,$bytes.Length)
    }finally{$stream.Dispose()}
}
if($FunctionsOnly){return}
if(-not $QueueDirectory){throw 'QueueDirectory required.'}
$QueueDirectory=(Resolve-Path -LiteralPath $QueueDirectory).Path
$project=Split-Path -Parent $PSScriptRoot
$task=Read-Json (Join-Path $QueueDirectory 'queue.json')
$powershell=Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'

if($ForceItem){
    $items=@($task.force_items|Where-Object {$_.name -ceq $ForceItem})
    if($items.Count -ne 1){throw 'Force item identity is absent or duplicated.'}
    $item=$items[0]
    $folder=Join-Path $QueueDirectory ('forces\'+$item.name)
    if(-not(Test-Path -LiteralPath (Join-Path $QueueDirectory ('claims\'+$item.name+'.json')))){throw 'Force worker requires an exclusive claim.'}
    New-Item -ItemType Directory -Path $folder -ErrorAction Stop | Out-Null
    $input=Join-Path $folder 'force.gjf'
    Copy-Item -LiteralPath (Join-Path $QueueDirectory $item.input) -Destination $input
    $timer=[Diagnostics.Stopwatch]::StartNew()
    $record=[ordered]@{kind='force';name=$item.name;reaction=$item.reaction;owner_pid=$PID;
        status='running';started_at_utc=[DateTime]::UtcNow.ToString('o');training_merge_authorized=$false}
    Write-Json (Join-Path $folder 'receipt.json') $record
    try{
        & (Join-Path $project 'run_gaussian_job.ps1') -InputFile $input -GaussianDirectory $GaussianDirectory|Out-Null
        $text=[IO.File]::ReadAllText((Join-Path $folder 'force.log'))
        if($text -notmatch 'Forces \(Hartrees/Bohr\)' -or $text -notmatch 'SCF Done:'){throw 'Missing Gaussian energy/force section.'}
        $record.status='raw_label_completed_pending_alignment'
    }catch{
        $record.status='failed';$record['error']=$_.Exception.ToString()
    }finally{
        $record['elapsed_seconds']=$timer.Elapsed.TotalSeconds
        $record['finished_at_utc']=[DateTime]::UtcNow.ToString('o')
        Write-Json (Join-Path $folder 'receipt.json') $record
    }
    return
}
if(-not $ExecutionFile){throw 'ExecutionFile required.'}
$ExecutionFile=(Resolve-Path -LiteralPath $ExecutionFile).Path
$execution=Read-Json $ExecutionFile
Assert-PoolSettings $execution
$stateFile=Join-Path $QueueDirectory 'queue-state.json'
$poolStartFile=Join-Path $QueueDirectory 'pool-start.json'
if(-not $Worker){
    if(Test-Path -LiteralPath $poolStartFile){throw 'Pool already submitted; inspect existing owner and evidence.'}
    if(-not $TakeoverOwner){throw 'Explicit existing serial owner is required for this dispatch.'}
    $state=Read-Json $stateFile
    if($state.owner_pid -ne $TakeoverOwner -or $state.status -ne 'running'){throw 'Serial owner/state does not match takeover.'}
    $owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+$TakeoverOwner)
    Assert-ChildIdentity $owner 'run_gaussian_queue.ps1' $QueueDirectory
    $children=@(Get-CimInstance Win32_Process|Where-Object {$_.ParentProcessId -eq $TakeoverOwner -and $_.Name -eq 'powershell.exe'})
    if($children.Count -ne 1){throw 'Takeover requires exactly one known foreground chain child.'}
    $child=$children[0]
    $live=@()
    foreach($item in $task.chains){
        $folder=Join-Path $QueueDirectory $item.directory
        $path=Join-Path $folder 'worker-state.json'
        if(Test-Path -LiteralPath $path){
            $record=Read-Json $path
            if($record.status -eq 'running'){
                if($record.owner_pid -ne $child.ProcessId){throw 'Running chain owner differs from serial child.'}
                Assert-ChildIdentity $child 'run_gaussian_chain.ps1' $folder
                $live+=@{job_id=$item.job_id;owner_pid=$child.ProcessId;directory=$item.directory;command_line=$child.CommandLine;started_at_utc=$record.started_at_utc}
            }
        }
    }
    if($live.Count -ne 1){throw 'No unique live scientific chain to adopt.'}
    $scientificLock=$null
    try{$scientificLock=[IO.File]::Open((Join-Path (Join-Path $QueueDirectory $live[0].directory) 'owner.lock'),[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)}catch [IO.IOException]{}
    if($scientificLock){$scientificLock.Dispose();throw 'Scientific child is not holding its expected lock.'}
    $receipt=[ordered]@{previous_owner=$TakeoverOwner;submitted_at_utc=[DateTime]::UtcNow.ToString('o');
        adopted=$live;previous_state=$state;execution=$execution;scientific_children_restarted=$false}
    $claim=[IO.File]::Open($poolStartFile,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    $claim.Dispose()
    Write-Json $poolStartFile $receipt
    # Stop only the serial scheduler; its verified chain/Gaussian processes survive.
    Stop-Process -Id $TakeoverOwner -ErrorAction Stop
    $command="`"$powershell`" -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -QueueDirectory `"$QueueDirectory`" -ExecutionFile `"$ExecutionFile`" -GaussianDirectory `"$GaussianDirectory`" -Worker"
    $startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
    $created=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command;CurrentDirectory=$project;ProcessStartupInformation=$startup}
    if($created.ReturnValue -ne 0){throw 'Pool owner failed; adopted scientific child remains running. Inspect before recovery.'}
    $receipt['new_owner_pid']=$created.ProcessId;Write-Json $poolStartFile $receipt
    [ordered]@{owner_pid=$created.ProcessId;adopted=$live;execution=$execution}|ConvertTo-Json -Depth 8
    return
}

$lock=[IO.File]::Open((Join-Path $QueueDirectory 'queue.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$receipt=Read-Json $poolStartFile
if(-not $receipt.adopted){throw 'Missing explicit adoption receipt.'}
if(Test-Path -LiteralPath (Join-Path $QueueDirectory 'pool-worker-start.json')){throw 'Pool worker already ran; no implicit recovery.'}
Write-Json (Join-Path $QueueDirectory 'pool-worker-start.json') @{owner_pid=$PID;started_at_utc=[DateTime]::UtcNow.ToString('o')}
$claims=Join-Path $QueueDirectory 'claims';New-Item -ItemType Directory -Path $claims -ErrorAction Stop|Out-Null
$state=[ordered]@{host=$env:COMPUTERNAME;owner_pid=$PID;started_at_utc=$receipt.previous_state.started_at_utc;
    pool_started_at_utc=[DateTime]::UtcNow.ToString('o');status='running';stage='adopting';
    outcomes=@($receipt.previous_state.outcomes);active=@();compute_seconds=[double]$receipt.previous_state.compute_seconds;
    execution=$execution;holdout_labels_evaluated=$false;training_merge_authorized=$false;automatic_retries=0}
$deadline=[DateTime]::UtcNow.AddHours($execution.wall_hours_from_pool_start)
$state['admission_deadline_utc']=$deadline.ToString('o')
$active=@{};$pending=[Collections.Generic.Queue[object]]::new();$failures=0;$admissionBlocked=$false
function Save-State{
    $state.updated_at_utc=[DateTime]::UtcNow.ToString('o');$state.active=@($active.Values)
    Write-Json $stateFile $state
}
try{
    foreach($item in $task.chains){
        $folder=Join-Path $QueueDirectory $item.directory;$path=Join-Path $folder 'worker-state.json'
        $inputTask=Read-Json (Join-Path $folder 'task.json')
        if($inputTask.configuration.dft_nproc -ne 4 -or $inputTask.configuration.dft_mem -ne '2GB'){throw 'Scientific input exceeds fixed per-worker resource envelope.'}
        $adopted=@($receipt.adopted|Where-Object {$_.job_id -eq $item.job_id})
        if($adopted.Count){
            $record=Read-Json $path;$child=Get-CimInstance Win32_Process -Filter ('ProcessId='+$record.owner_pid)
            Assert-ChildIdentity $child 'run_gaussian_chain.ps1' $folder
            Claim-Item $claims $item.job_id
            $active[$item.job_id]=@{kind='chain';name=$item.job_id;directory=$item.directory;owner_pid=$record.owner_pid;command_line=$child.CommandLine;started_at_utc=$record.started_at_utc;adopted=$true}
        }elseif(Test-Path -LiteralPath $path){
            $record=Read-Json $path
            if($record.status -eq 'running'){throw 'Unowned running chain.'}
            Claim-Item $claims $item.job_id
            if(-not @($state.outcomes|Where-Object {$_.job_id -eq $item.job_id}).Count){
                $state.outcomes+=@{kind='chain';job_id=$item.job_id;status=$record.status;directory=$item.directory;elapsed_seconds=$record.elapsed_seconds}
                $state.compute_seconds+=$record.elapsed_seconds
            }
        }else{$pending.Enqueue(@{kind='chain';name=$item.job_id;directory=$item.directory})}
    }
    foreach($item in $task.force_items){
        if(Test-Path -LiteralPath (Join-Path $QueueDirectory ('forces\'+$item.name))){throw 'Existing force evidence requires diagnosis before pool takeover.'}
        $pending.Enqueue(@{kind='force';name=$item.name;directory=('forces/'+$item.name)})
    }
    Save-State
    while($pending.Count -or $active.Count){
        $processes=@(Get-CimInstance Win32_Process)
        foreach($name in @($active.Keys)){
            $item=$active[$name];$live=@($processes|Where-Object {$_.ProcessId -eq $item.owner_pid})
            if($live.Count){if($live[0].CommandLine -cne $item.command_line){throw 'Child PID changed identity.'};continue}
            $folder=Join-Path $QueueDirectory $item.directory
            $path=if($item.kind -eq 'chain'){Join-Path $folder 'worker-state.json'}else{Join-Path $folder 'receipt.json'}
            $record=if(Test-Path -LiteralPath $path){Read-Json $path}else{$null}
            if(-not $record -or $record.status -eq 'running'){throw ('Child missing without terminal evidence: '+$name)}
            $elapsed=if($null -ne $record.elapsed_seconds){[double]$record.elapsed_seconds}else{([DateTime]::UtcNow-[DateTime]$item.started_at_utc).TotalSeconds}
            $state.outcomes+=@{kind=$item.kind;job_id=$name;name=$name;status=$record.status;directory=$item.directory;elapsed_seconds=$elapsed}
            $state.compute_seconds+=$elapsed
            if($record.status -eq 'failed'){$failures++}else{$failures=0}
            $active.Remove($name)
        }
        if($failures -ge $execution.failure_limit){$admissionBlocked=$true;$state['admission_blocked_reason']='Three consecutive failures; drain active workers without new admission.'}
        $os=Get-CimInstance Win32_OperatingSystem;$free=$os.FreePhysicalMemory/1MB
        $state['available_GiB']=$free
        if($free -lt $execution.minimum_available_GiB_to_keep_admission){$state['memory_pressure_observed_utc']=[DateTime]::UtcNow.ToString('o')}
        # Every observed g16 must be owned by one of our exact native children.
        $g16=@($processes|Where-Object {$_.Name -eq 'g16.exe'})
        $knownParents=@($active.Values|ForEach-Object {$_.owner_pid})
        if(@($g16|Where-Object {$_.ParentProcessId -notin $knownParents}).Count){$admissionBlocked=$true;$state['admission_blocked_reason']='Unknown Gaussian owner present; drain known workers.'}
        if($pending.Count -and (Can-Admit $execution $active.Count $free $deadline ([DateTime]::UtcNow) $admissionBlocked)){
            $item=$pending.Dequeue();Claim-Item $claims $item.name
            $folder=Join-Path $QueueDirectory $item.directory
            if($item.kind -eq 'chain'){
                $script=Join-Path $PSScriptRoot 'run_gaussian_chain.ps1'
                $arguments="-NoProfile -ExecutionPolicy Bypass -File `"$script`" -JobDirectory `"$folder`" -GaussianDirectory `"$GaussianDirectory`" -Worker"
            }else{
                $script=$PSCommandPath
                $arguments="-NoProfile -ExecutionPolicy Bypass -File `"$script`" -QueueDirectory `"$QueueDirectory`" -GaussianDirectory `"$GaussianDirectory`" -ForceItem $($item.name)"
            }
            $child=Start-Process -FilePath $powershell -ArgumentList $arguments -WindowStyle Hidden -WorkingDirectory $project -PassThru
            $identity=Get-CimInstance Win32_Process -Filter ('ProcessId='+$child.Id)
            if(-not $identity){throw 'Launched native worker exited before identity capture; inspect claim/log before recovery.'}
            $active[$item.name]=@{kind=$item.kind;name=$item.name;directory=$item.directory;owner_pid=$child.Id;command_line=$identity.CommandLine;started_at_utc=[DateTime]::UtcNow.ToString('o');adopted=$false}
        }
        $state.stage=if($active.Count){(@($active.Keys)|Sort-Object)-join ','}else{'waiting_for_memory'}
        $state['pending_items']=$pending.Count;Save-State
        if(($admissionBlocked -or [DateTime]::UtcNow -ge $deadline) -and $active.Count -eq 0){break}
        if($pending.Count -or $active.Count){Start-Sleep -Seconds $execution.poll_seconds}
    }
    $state.status=if($admissionBlocked){'blocked_after_draining'}elseif($pending.Count){'wall_budget_reached'}else{'prepared_pool_exhausted'}
}catch{
    $state.status='blocked';$state['error']=$_.Exception.ToString();$state['scientific_children_left_running_for_diagnosis']=$true
}finally{
    $state['finished_at_utc']=[DateTime]::UtcNow.ToString('o');Save-State;$lock.Dispose()
}
