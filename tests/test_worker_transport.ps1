param([string]$ProjectRoot=(Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference='Stop'
. (Join-Path $ProjectRoot 'scripts\worker_transport.ps1')
$directory=Join-Path ([IO.Path]::GetTempPath()) ('.agents\ai4chem-transport-test-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $directory -ErrorAction Stop | Out-Null
$fixture=Join-Path $directory 'fixture.ps1'
$powershell=Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
try {
    [IO.File]::WriteAllText($fixture,@'
param([string]$Mode)
if($Mode -eq 'flood') {
    [Console]::Out.Write(('x'*1048576))
    [Console]::Error.Write(('e'*1048576))
} elseif($Mode -eq 'fail') {
    [Console]::Error.Write('explicit-failure')
    exit 7
} elseif($Mode -eq 'sleep') {
    Start-Sleep -Seconds 10
}
'@,[Text.UTF8Encoding]::new($false))
    $arguments=@('-NoProfile','-File',$fixture,'-Mode','flood')
    $value=Invoke-WorkerCommand $powershell $arguments 15000
    if($value.Length -ne 1048576 -or $value -ne ('x'*1048576)) { throw 'Concurrent stream capture lost output.' }
    $failed=$false
    try { [void](Invoke-WorkerCommand $powershell @('-NoProfile','-File',$fixture,'-Mode','fail') 15000) }
    catch { $failed=$_.Exception.Message.Contains('explicit-failure') -and $_.Exception.Message.Contains('(7)') }
    if(-not $failed) { throw 'Native command failure was not propagated.' }
    $timedOut=$false
    $watch=[Diagnostics.Stopwatch]::StartNew()
    try { [void](Invoke-WorkerCommand $powershell @('-NoProfile','-File',$fixture,'-Mode','sleep') 500) }
    catch { $timedOut=$_.Exception.Message.Contains('finite timeout') }
    if(-not $timedOut -or $watch.Elapsed.TotalSeconds -ge 5) { throw 'Timeout did not terminate the transport promptly.' }
    Write-Output 'PASS: large stdout/stderr, nonzero exit and bounded transport timeout.'
} finally {
    Remove-Item -LiteralPath $directory -Recurse -Force
}
