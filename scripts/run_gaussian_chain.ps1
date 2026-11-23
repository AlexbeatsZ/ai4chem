param(
    [Parameter(Mandatory=$true)][string]$JobDirectory,
    [string]$GaussianDirectory = 'C:\G16W',
    [switch]$Worker,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$JobDirectory = (Resolve-Path -LiteralPath $JobDirectory).Path
$taskFile = Join-Path $JobDirectory 'task.json'
$task = [IO.File]::ReadAllText($taskFile) | ConvertFrom-Json
$stateFile = Join-Path $JobDirectory 'worker-state.json'
$project = Split-Path -Parent $PSScriptRoot
if (-not $Worker) {
    if (Test-Path -LiteralPath $stateFile) {
        if (-not $Resume) { throw 'Existing worker state requires diagnosis and explicit -Resume.' }
        $previous = [IO.File]::ReadAllText($stateFile) | ConvertFrom-Json
        if ($previous.status -eq 'running' -or $previous.raw_chain_completed -or
            (Get-Process -Id $previous.owner_pid -ErrorAction SilentlyContinue)) { throw 'Previous worker is active or already completed.' }
    } elseif ($Resume) { throw 'No previous state to resume.' }
    if (-not (Test-Path -LiteralPath (Join-Path $GaussianDirectory 'g16.exe'))) { throw 'Gaussian executable missing.' }
    $os = Get-CimInstance Win32_OperatingSystem
    if ($os.FreePhysicalMemory -lt 6MB) { throw 'Require at least 6 GiB available physical memory for this 4 GB job.' }
    $active = @(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^(g16|l[0-9]+)\.exe$'})
    if ($active.Count) { throw 'Another native Gaussian job is present; inspect before sharing this worker.' }
    $powershell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $command = "`"$powershell`" -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -JobDirectory `"$JobDirectory`" -GaussianDirectory `"$GaussianDirectory`" -Worker"
    if ($Resume) {
        $archive = Join-Path $JobDirectory ('previous-worker-state-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')+'.json')
        Copy-Item -LiteralPath $stateFile -Destination $archive
        $command += ' -Resume'
    }
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
    $created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine=$command;CurrentDirectory=$project;ProcessStartupInformation=$startup
    }
    if ($created.ReturnValue -ne 0) { throw "Independent launcher failed: $($created.ReturnValue)" }
    [ordered]@{host=$env:COMPUTERNAME;owner_pid=$created.ProcessId;job_id=$task.job_id;
        started_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
    return
}
$lock = [IO.File]::Open((Join-Path $JobDirectory 'owner.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$timer = [Diagnostics.Stopwatch]::StartNew()
$state = [ordered]@{job_id=$task.job_id;reaction=$task.reaction;host=$env:COMPUTERNAME;
    owner_pid=$PID;started_at_utc=[DateTime]::UtcNow.ToString('o');status='running';stage='starting';
    gaussian_directory=$GaussianDirectory;scientific_configuration=$task.configuration;
    stages=@();branches=@();raw_chain_completed=$false;dft_verified=$false}
function Save-State {
    $state.updated_at_utc = [DateTime]::UtcNow.ToString('o')
    $state.elapsed_seconds = $timer.Elapsed.TotalSeconds
    $temporary = Join-Path $JobDirectory 'worker-state.tmp'
    [IO.File]::WriteAllText($temporary,($state | ConvertTo-Json -Depth 15),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $stateFile -Force
}
function Read-Stage([string]$name) {
    $text = [IO.File]::ReadAllText((Join-Path $JobDirectory "$name.log"))
    $section = ($text -split 'Harmonic frequencies')[-1]
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $frequencies = @([regex]::Matches($section,'Frequencies\s+--([^\r\n]+)') | ForEach-Object {
        [regex]::Matches($_.Groups[1].Value,'-?\d+\.\d+') | ForEach-Object {[double]::Parse($_.Value,$culture)}
    })
    $energyMatches = [regex]::Matches($text,'SCF Done:.*?=\s*([-\d.]+)')
    $energy = $null
    if ($energyMatches.Count) { $energy = [double]::Parse($energyMatches[$energyMatches.Count-1].Groups[1].Value,$culture) }
    $minimum = $text.Contains('PES minimum detected')
    $complete = $text -match 'Calculation of (FORWARD|REVERSE) path complete'
    $postError = $text.Contains('PESDtA: Invalid IPoint.') -and $minimum -and $complete -and
        ($text.LastIndexOf('PES minimum detected') -lt $text.LastIndexOf('Reaction path calculation complete.')) -and
        ($text.LastIndexOf('Reaction path calculation complete.') -lt $text.LastIndexOf('PESDtA: Invalid IPoint.')) -and
        ([regex]::Matches($text,'Error termination').Count -eq 1)
    [ordered]@{job=$name;normal_termination=($text.TrimEnd() -match 'Normal termination of Gaussian[^\r\n]*\z');
        optimization_completed=$text.Contains('Optimization completed.');frequencies_cm1=$frequencies;
        imaginary_modes=@($frequencies | Where-Object {$_ -lt 0}).Count;
        significant_imaginary_modes=@($frequencies | Where-Object {$_ -lt -50}).Count;
        energy_hartree=$energy;minimum_detected=$minimum;direction_complete=$complete;
        maximum_steps_reached=$text.Contains('Maximum number of steps reached.');
        postprocessing_error=$postError;recovered_postprocessing_error=$false;
        irc_points=[regex]::Matches($text,'NET REACTION COORDINATE UP TO THIS POINT').Count}
}
function Run-Stage([string]$name,[bool]$allowPostprocessingError=$false) {
    $state.stage = $name
    Save-State
    $log = Join-Path $JobDirectory "$name.log"
    if (Test-Path -LiteralPath $log) {
        if (-not $Resume) { throw "Existing log requires diagnosis: $log" }
        $record = Read-Stage $name
        if (-not ($record.normal_termination -or ($allowPostprocessingError -and $record.postprocessing_error))) {
            throw "Incomplete/failed stage requires diagnosis; refusing overwrite: $log"
        }
        $record.elapsed_seconds = 0
        $record.reused_existing_stage = $true
        if (-not $record.normal_termination) { $record.recovered_postprocessing_error = $true }
        $state.stages += $record
        Save-State
        return $record
    }
    $clock = [Diagnostics.Stopwatch]::StartNew()
    try {
        & (Join-Path $project 'run_gaussian_job.ps1') -InputFile (Join-Path $JobDirectory "$name.gjf") -GaussianDirectory $GaussianDirectory | Out-Null
    } catch {
        if (-not (Test-Path -LiteralPath $log)) { throw }
        $failed = Read-Stage $name
        if (-not ($allowPostprocessingError -and $failed.postprocessing_error -and
                  (Test-Path -LiteralPath (Join-Path $JobDirectory "$name.chk")))) { throw }
    }
    $record = Read-Stage $name
    $record.elapsed_seconds = $clock.Elapsed.TotalSeconds
    if (-not $record.normal_termination) { $record.recovered_postprocessing_error = $true }
    $state.stages += $record
    Save-State
    return $record
}
try {
    Save-State
    $ts = Run-Stage 'ts'
    if (-not ($ts.optimization_completed -and $ts.frequencies_cm1.Count -eq $task.expected_vibrational_modes -and
              $ts.imaginary_modes -eq 1 -and $ts.significant_imaginary_modes -eq 1)) {
        throw 'TS failed fixed optimization/frequency acceptance.'
    }
    foreach ($direction in @('forward','reverse')) {
        $name = "irc_$direction"
        $irc = Run-Stage $name $true
        if (-not ($irc.minimum_detected -and $irc.direction_complete)) {
            if (-not $irc.maximum_steps_reached) { throw "IRC did not reach a minimum: $name" }
            $policy = $task.configuration.dft_policy
            if ($policy.irc_initial_max_points -and
                $policy.irc_initial_max_points -ge $policy.irc_restart_max_points) {
                throw "IRC exhausted its fixed budget: $name"
            }
            $name = "${name}_extended"
            $irc = Run-Stage $name $true
        }
        if (-not ($irc.minimum_detected -and $irc.direction_complete)) { throw "IRC exhausted its fixed budget: $name" }
        $minimumName = "minimum_$direction"
        if ($name.EndsWith('_extended')) { $minimumName += '_extended' }
        $minimum = Run-Stage $minimumName
        if (-not ($minimum.optimization_completed -and $minimum.frequencies_cm1.Count -eq $task.expected_vibrational_modes -and
                  $minimum.imaginary_modes -eq 0)) { throw "Unstable or incomplete minimum: $minimumName" }
        $state.branches += [ordered]@{direction=$direction;irc_job=$name;minimum_job=$minimumName}
        Save-State
    }
    $state.status = 'raw_chain_completed'
    $state.stage = 'awaiting_geometry_validation'
    $state.raw_chain_completed = $true
} catch {
    $state.status = 'failed'
    $state.error = $_.Exception.ToString()
    $state.script_stack_trace = $_.ScriptStackTrace
} finally {
    $state.finished_at_utc = [DateTime]::UtcNow.ToString('o')
    Save-State
    $lock.Dispose()
}
