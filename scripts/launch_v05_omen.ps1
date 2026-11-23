param(
    [ValidateSet('coordinator','ready-dft','dft')][string]$Mode = 'coordinator',
    [switch]$RetryFailed,
    [switch]$Worker
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$project = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $project 'results\coordination'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$shellScript = 'run_v05_omen.sh'
if($Mode -eq 'ready-dft') { $shellScript = 'run_ready_dft_omen.sh' }
if($Mode -eq 'dft') { $shellScript = 'run_dft_omen.sh' }
$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'wsl.exe' -and $_.CommandLine -like "*$shellScript*"
}
if($active) { throw "A Windows WSL launcher for $Mode is already running" }
if($RetryFailed -and $Mode -ne 'ready-dft') { throw 'RetryFailed is only valid for ready-dft' }
if(-not $Worker) {
    # WMI owns the process instead of the short-lived OpenSSH job. No scheduled
    # task, global environment setting or startup registration is created.
    $powershell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $command = "`"$powershell`" -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Mode $Mode -Worker"
    if($RetryFailed) { $command += ' -RetryFailed' }
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
    $created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine=$command;CurrentDirectory=$project;ProcessStartupInformation=$startup
    }
    if($created.ReturnValue -ne 0) { throw "Independent launcher failed: $($created.ReturnValue)" }
    [ordered]@{host='OMEN';mode=$Mode;windows_owner_pid=$created.ProcessId;
        started_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
    return
}
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$log = Join-Path $runtime "$Mode-$stamp.log"
$errorLog = Join-Path $runtime "$Mode-$stamp.err"
$arguments = @('-d','Ubuntu-24.04','--','bash',"/mnt/c/Users/Meta/Project/Workspaces/ai4chem/scripts/$shellScript")
if($RetryFailed) { $arguments += '--retry-failed' }
# Keep this native wsl.exe alive for the full Linux task so Windows interop
# retains its owning session, including Gaussian launched hours later.
$process = Start-Process -FilePath (Join-Path $env:WINDIR 'System32\wsl.exe') `
    -ArgumentList $arguments -WorkingDirectory $project -WindowStyle Hidden `
    -RedirectStandardOutput $log -RedirectStandardError $errorLog -PassThru
$record = [ordered]@{host='OMEN';mode=$Mode;windows_owner_pid=$PID;windows_launcher_pid=$process.Id;
    started_at_utc=[DateTime]::UtcNow.ToString('o');log=$log;error_log=$errorLog}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtime "$Mode-launch.json") -Encoding UTF8
$record | ConvertTo-Json
Wait-Process -Id $process.Id
