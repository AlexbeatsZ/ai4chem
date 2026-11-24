param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$ValidationId,
    [ValidateSet("ml","install")][string]$Mode="ml",
    [switch]$Worker
)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$directory=Join-Path $project ('results\coordination\cloud-validation\'+$ValidationId)
if (-not $Worker) {
    if (Test-Path -LiteralPath $directory) { throw 'Validation ID already exists; inspect evidence instead of overwriting.' }
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $powershell=Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $command="`"$powershell`" -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -ValidationId $ValidationId -Mode $Mode -Worker"
    $startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
    $created=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command;CurrentDirectory=$project;ProcessStartupInformation=$startup}
    if ($created.ReturnValue -ne 0) { throw 'Independent validation owner failed.' }
    [ordered]@{host=$env:COMPUTERNAME;owner_pid=$created.ProcessId;validation_id=$ValidationId;started_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
    return
}
$lock=[IO.File]::Open((Join-Path $directory 'owner.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$linuxRoot=(& wsl.exe -d Ubuntu-24.04 --exec wslpath -a $project.Replace('\','/')).Trim()
if ($LASTEXITCODE -ne 0 -or -not $linuxRoot.StartsWith('/')) { throw 'Cannot resolve actual WSL checkout.' }
$stdout=Join-Path $directory 'stdout.log'
$stderr=Join-Path $directory 'stderr.log'
$process=Start-Process -FilePath (Join-Path $env:WINDIR 'System32\wsl.exe') -ArgumentList @('-d','Ubuntu-24.04','--exec','bash',"$linuxRoot/scripts/validate_cloud_omen.sh",$ValidationId,$Mode) -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$record=[ordered]@{host=$env:COMPUTERNAME;owner_pid=$PID;wsl_pid=$process.Id;validation_id=$ValidationId;started_at_utc=[DateTime]::UtcNow.ToString('o');stdout=$stdout;stderr=$stderr;stage='running'}
$state=Join-Path $directory 'launch.json'
[IO.File]::WriteAllText($state,($record|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
$process.WaitForExit()
$exitFile=Join-Path $project ('results\node-readiness\cloud-worker\'+$ValidationId+'\exit-code.txt')
if (Test-Path -LiteralPath $exitFile) { $record['script_exit_code']=[int]([IO.File]::ReadAllText($exitFile).Trim()) }
else { $record['script_exit_code']=$null }
$record['stage']='finished'
$record['finished_at_utc']=[DateTime]::UtcNow.ToString('o')
[IO.File]::WriteAllText($state,($record|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
