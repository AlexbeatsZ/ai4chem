param([Parameter(Mandatory=$true)][string]$Protocol,[switch]$Worker)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$Protocol=(Resolve-Path -LiteralPath $Protocol).Path
$output=Join-Path $project 'results\overnight-20261124\omen'
if(-not $Worker) {
    if(Test-Path -LiteralPath $output){throw 'Queue output already exists; inspect without restarting.'}
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    $powershell=Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $command="`"$powershell`" -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Protocol `"$Protocol`" -Worker"
    $startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
    $created=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command;CurrentDirectory=$project;ProcessStartupInformation=$startup}
    if($created.ReturnValue -ne 0){throw 'Native owner launch failed.'}
    [ordered]@{host=$env:COMPUTERNAME;owner_pid=$created.ProcessId;output=$output;started_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
    return
}
$linuxRoot=(& wsl.exe -d Ubuntu-24.04 --exec wslpath -a $project.Replace('\','/')).Trim()
$linuxProtocol=(& wsl.exe -d Ubuntu-24.04 --exec wslpath -a $Protocol.Replace('\','/')).Trim()
if($LASTEXITCODE -ne 0 -or -not $linuxRoot.StartsWith('/')){throw 'WSL path resolution failed.'}
$arguments=@('-d','Ubuntu-24.04','--exec','env',"PYTHONPATH=$linuxRoot",'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4',
    '/home/meta/.local/opt/miniforge3/bin/mamba','run','-n','ai4chem','python','-u',
    ($linuxRoot+'/scripts/run_paired_queue.py'),'--protocol',$linuxProtocol,'--node','omen')
$quoted=($arguments | ForEach-Object {if($_ -match '\s'){'"'+$_+'"'}else{$_}}) -join ' '
$process=Start-Process -FilePath (Join-Path $env:WINDIR 'System32\wsl.exe') -ArgumentList $quoted -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput (Join-Path $output 'launcher.log') -RedirectStandardError (Join-Path $output 'launcher-error.log') -PassThru
$record=[ordered]@{owner_pid=$PID;wsl_pid=$process.Id;started_at_utc=[DateTime]::UtcNow.ToString('o');status='running'}
$state=Join-Path $output 'native-owner.json'
[IO.File]::WriteAllText($state,($record|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
$process.WaitForExit()
$record.status='finished';$record['finished_at_utc']=[DateTime]::UtcNow.ToString('o')
[IO.File]::WriteAllText($state,($record|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
