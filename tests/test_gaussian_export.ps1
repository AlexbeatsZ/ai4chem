param([Parameter(Mandatory=$true)][string]$TemporaryRoot)
$ErrorActionPreference='Stop'
$directory=Join-Path $TemporaryRoot 'terminal-job'
New-Item -ItemType Directory -Path $directory -ErrorAction Stop | Out-Null
$exporter=Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts\export_gaussian_chain.ps1'
$statePath=Join-Path $directory 'worker-state.json'
$state='{"status":"raw_chain_completed","owner_pid":2147483647,"job_id":"r1"}'
[IO.File]::WriteAllText($statePath,$state)
[IO.File]::WriteAllText((Join-Path $directory 'ts.log'),'terminal-log')
$first=(& $exporter -JobDirectory $directory | ConvertFrom-Json)
$stamp=(Get-Item -LiteralPath $first.archive).LastWriteTimeUtc
$second=(& $exporter -JobDirectory $directory | ConvertFrom-Json)
if($first.bytes -ne $second.bytes -or (Get-Item -LiteralPath $first.archive).LastWriteTimeUtc -ne $stamp) {
    throw 'Identical terminal export was overwritten.'
}
$rejectedIdentity=$false
[IO.File]::WriteAllText($statePath,$state.Replace('"r1"','"r2"'))
try { & $exporter -JobDirectory $directory | Out-Null } catch { $rejectedIdentity=$_.Exception.Message -like '*identity differs*' }
if(-not $rejectedIdentity) { throw 'Same-length changed job identity was accepted.' }
[IO.File]::WriteAllText($statePath,$state)
$rejectedLength=$false
[IO.File]::WriteAllText((Join-Path $directory 'ts.log'),'changed-terminal-log')
try { & $exporter -JobDirectory $directory | Out-Null } catch { $rejectedLength=$_.Exception.Message -like '*entry differs*' }
if(-not $rejectedLength) { throw 'Changed source evidence was accepted.' }
if((Get-Item -LiteralPath $first.archive).LastWriteTimeUtc -ne $stamp) { throw 'Failed reuse replaced evidence.' }
[ordered]@{status='PASS';identical_export_reused=$true;changed_identity_rejected=$true;changed_log_rejected=$true;archive_preserved=$true} | ConvertTo-Json
