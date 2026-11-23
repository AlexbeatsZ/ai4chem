param(
    [ValidateSet('download','study')][string]$Stage = 'study',
    [switch]$Worker
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $project 'results\coordination\model-v07'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
if(-not $Worker) {
    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'powershell.exe' -and $_.CommandLine -like '*launch_model_study.ps1*' -and
        $_.CommandLine -like "*-Stage $Stage -Worker*"
    }
    if($active) { throw "Model study stage $Stage already has an owner" }
    $powershell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $command = "`"$powershell`" -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Stage $Stage -Worker"
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
    $created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine=$command;CurrentDirectory=$project;ProcessStartupInformation=$startup
    }
    if($created.ReturnValue -ne 0) { throw "Independent owner failed: $($created.ReturnValue)" }
    [ordered]@{host='OMEN';stage=$Stage;owner_pid=$created.ProcessId;
        started_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
    return
}
$linuxRoot = (& wsl.exe -d Ubuntu-24.04 -- wslpath -a $project.Replace('\','/')).Trim()
if($LASTEXITCODE -ne 0 -or -not $linuxRoot.StartsWith('/')) { throw 'wslpath failed' }
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$log = Join-Path $runtime "$Stage-$stamp.log"
$errorLog = Join-Path $runtime "$Stage-$stamp.err"
$process = Start-Process -FilePath (Join-Path $env:WINDIR 'System32\wsl.exe') `
    -ArgumentList @('-d','Ubuntu-24.04','--','bash',"$linuxRoot/scripts/run_model_study.sh",$Stage) `
    -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput $log `
    -RedirectStandardError $errorLog -PassThru
$record = [ordered]@{host='OMEN';stage=$Stage;owner_pid=$PID;wsl_pid=$process.Id;
    started_at_utc=[DateTime]::UtcNow.ToString('o');log=$log;error_log=$errorLog}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtime "$Stage-launch.json") -Encoding UTF8
Wait-Process -Id $process.Id
$process.Refresh()
$record['exit_code'] = $process.ExitCode
$record['finished_at_utc'] = [DateTime]::UtcNow.ToString('o')
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtime "$Stage-launch.json") -Encoding UTF8
