param([Parameter(Mandatory=$true)][string]$QueueDirectory,[string]$GaussianDirectory='C:\G16W',[switch]$Worker)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$QueueDirectory=(Resolve-Path -LiteralPath $QueueDirectory).Path
$project=Split-Path -Parent $PSScriptRoot
$task=[IO.File]::ReadAllText((Join-Path $QueueDirectory 'queue.json')) | ConvertFrom-Json
$stateFile=Join-Path $QueueDirectory 'queue-state.json'
if(-not $Worker) {
    if(Test-Path -LiteralPath $stateFile){throw 'Existing queue evidence; no implicit restart.'}
    if(-not(Test-Path -LiteralPath (Join-Path $GaussianDirectory 'g16.exe'))){throw 'Gaussian missing.'}
    $powershell=Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $command="`"$powershell`" -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -QueueDirectory `"$QueueDirectory`" -GaussianDirectory `"$GaussianDirectory`" -Worker"
    $startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
    $created=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command;CurrentDirectory=$project;ProcessStartupInformation=$startup}
    if($created.ReturnValue -ne 0){throw 'Native queue owner failed.'}
    [ordered]@{host=$env:COMPUTERNAME;owner_pid=$created.ProcessId;queue=$QueueDirectory;started_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
    return
}
$lock=[IO.File]::Open((Join-Path $QueueDirectory 'queue.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
if(Test-Path -LiteralPath $stateFile){$lock.Dispose();throw 'Existing queue evidence; no implicit restart.'}
$state=[ordered]@{host=$env:COMPUTERNAME;owner_pid=$PID;started_at_utc=[DateTime]::UtcNow.ToString('o');status='running';stage='admission';outcomes=@();compute_seconds=0.0;holdout_labels_evaluated=$false;training_merge_authorized=$false;automatic_retries=0}
function Save-State {
    $state.updated_at_utc=[DateTime]::UtcNow.ToString('o')
    $temporary=Join-Path $QueueDirectory 'queue-state.tmp'
    [IO.File]::WriteAllText($temporary,($state|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $stateFile -Force
}
function Admit {
    while($true) {
        $os=Get-CimInstance Win32_OperatingSystem
        $active=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^(g16|l[0-9]+)\.exe$'})
        if($active.Count){throw 'Unexpected Gaussian process; stop instead of overlapping work.'}
        if($os.FreePhysicalMemory -ge 4MB){return}
        $state.stage='waiting_for_4GiB_free';Save-State
        Start-Sleep -Seconds 60
    }
}
try {
    Save-State
    foreach($item in $task.chains) {
        Admit
        $folder=Join-Path $QueueDirectory $item.directory
        if(Test-Path -LiteralPath (Join-Path $folder 'worker-state.json')){throw 'Existing chain output; inspect without overwrite.'}
        $state.stage=$item.job_id;Save-State
        $timer=[Diagnostics.Stopwatch]::StartNew()
        # A distinct foreground native child gives each finished chain a terminal
        # owner PID, allowing evidence export while this queue keeps computing.
        $powershell=Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
        & $powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'run_gaussian_chain.ps1') -JobDirectory $folder -GaussianDirectory $GaussianDirectory -Worker
        if($LASTEXITCODE -ne 0){throw 'Chain child failed before writing a terminal record.'}
        $record=[IO.File]::ReadAllText((Join-Path $folder 'worker-state.json')) | ConvertFrom-Json
        $state.compute_seconds+=$timer.Elapsed.TotalSeconds
        $state.outcomes+= [ordered]@{kind='chain';job_id=$item.job_id;status=$record.status;directory=$item.directory;elapsed_seconds=$timer.Elapsed.TotalSeconds}
        Save-State
    }
    $consecutiveFailures=0
    foreach($item in $task.force_items) {
        if($state.compute_seconds -ge $task.target_compute_hours*3600){break}
        Admit
        $folder=Join-Path $QueueDirectory ('forces\'+$item.name)
        New-Item -ItemType Directory -Path $folder -ErrorAction Stop | Out-Null
        $input=Join-Path $folder 'force.gjf'
        Copy-Item -LiteralPath (Join-Path $QueueDirectory $item.input) -Destination $input
        $state.stage=$item.name;Save-State
        $timer=[Diagnostics.Stopwatch]::StartNew()
        $errorText=$null
        try {
            & (Join-Path $project 'run_gaussian_job.ps1') -InputFile $input -GaussianDirectory $GaussianDirectory | Out-Null
            $text=[IO.File]::ReadAllText((Join-Path $folder 'force.log'))
            if($text -notmatch 'Forces \(Hartrees/Bohr\)' -or $text -notmatch 'SCF Done:'){throw 'Missing Gaussian energy/force section.'}
            $status='raw_label_completed_pending_alignment';$consecutiveFailures=0
        } catch {
            $status='failed';$errorText=$_.Exception.ToString();$consecutiveFailures++
        }
        $elapsed=$timer.Elapsed.TotalSeconds
        $state.compute_seconds+=$elapsed
        $record=[ordered]@{kind='force';name=$item.name;reaction=$item.reaction;status=$status;error=$errorText;elapsed_seconds=$elapsed;training_merge_authorized=$false}
        [IO.File]::WriteAllText((Join-Path $folder 'receipt.json'),($record|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
        $state.outcomes+=$record;Save-State
        if($consecutiveFailures -ge 3){throw 'Three consecutive force failures; stop and retain evidence for diagnosis.'}
    }
    $state.status=if($state.compute_seconds -ge $task.target_compute_hours*3600){'compute_budget_reached'}else{'prepared_pool_exhausted'}
} catch {
    $state.status='blocked';$state['error']=$_.Exception.ToString()
} finally {
    $state['finished_at_utc']=[DateTime]::UtcNow.ToString('o');Save-State;$lock.Dispose()
}
