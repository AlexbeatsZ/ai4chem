param(
    [Parameter(Mandatory=$true)][string]$Bundle,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$LinuxPrefix='',
    [switch]$Worker
)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$Bundle=[IO.Path]::GetFullPath($Bundle)
$Output=[IO.Path]::GetFullPath($Output)
foreach ($value in @($Bundle,$Output,$LinuxPrefix,$PSCommandPath)) {
    if ($value.Contains('"') -or $value.Contains("`n") -or $value.Contains("`r")) { throw 'Unsupported quote/newline in path.' }
}
if (-not (Test-Path -LiteralPath (Join-Path $Bundle 'task.json'))) { throw 'Bundle task.json missing.' }
if (-not $Worker) {
    if ((Test-Path -LiteralPath $Output) -or (Test-Path -LiteralPath ($Output+'.tar.gz'))) { throw 'Use a fresh output path; retain previous evidence.' }
    New-Item -ItemType Directory -Path $Output -Force | Out-Null
    $powershell=Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $command="`"$powershell`" -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Bundle `"$Bundle`" -Output `"$Output`" -Worker"
    if ($LinuxPrefix) { $command+=" -LinuxPrefix `"$LinuxPrefix`"" }
    $startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
    $created=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command;CurrentDirectory=$project;ProcessStartupInformation=$startup}
    if ($created.ReturnValue -ne 0) { throw 'Independent Windows owner failed.' }
    [ordered]@{host=$env:COMPUTERNAME;owner_pid=$created.ProcessId;output=$Output;started_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
    return
}
function LinuxPath([string]$path) {
    $result=(& wsl.exe -d Ubuntu-24.04 --exec wslpath -a $path.Replace('\','/')).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $result.StartsWith('/')) { throw 'Cannot resolve WSL path.' }
    return $result
}
$linuxRoot=LinuxPath $project
$linuxBundle=LinuxPath $Bundle
$linuxOutput=LinuxPath $Output
if (-not $LinuxPrefix) { $LinuxPrefix=$linuxRoot+'/environments/cloud-worker/.runtime/env' }
if (-not $LinuxPrefix.StartsWith('/')) { throw 'Dedicated prefix must be absolute.' }
$arguments=@('-d','Ubuntu-24.04','--exec','bash',($linuxRoot+'/scripts/run_cloud_training_bundle.sh'),'--worker',$linuxBundle,$linuxOutput,$LinuxPrefix)
$quoted=($arguments | ForEach-Object { if ($_ -match '\s') { '"'+$_+'"' } else { $_ } }) -join ' '
$process=Start-Process -FilePath (Join-Path $env:WINDIR 'System32\wsl.exe') -ArgumentList $quoted -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Output 'launcher.log') -RedirectStandardError (Join-Path $Output 'launcher-error.log') -PassThru
$record=[ordered]@{host=$env:COMPUTERNAME;owner_pid=$PID;wsl_pid=$process.Id;started_at_utc=[DateTime]::UtcNow.ToString('o');stage='running';bundle=$Bundle;output=$Output;linux_prefix=$LinuxPrefix}
$state=Join-Path $Output 'native-owner.json'
[IO.File]::WriteAllText($state,($record|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
$process.WaitForExit()
$record['stage']='finished'
$record['finished_at_utc']=[DateTime]::UtcNow.ToString('o')
$exitFile=Join-Path $Output 'exit-code.txt'
if (Test-Path -LiteralPath $exitFile) { $record['script_exit_code']=[int]([IO.File]::ReadAllText($exitFile).Trim()) }
else { $record['script_exit_code']=$null }
[IO.File]::WriteAllText($state,($record|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
