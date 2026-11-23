param(
    [Parameter(Mandatory=$true)][string]$JobDirectory,
    [switch]$RemoveArchive
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$JobDirectory = (Resolve-Path -LiteralPath $JobDirectory).Path
$archive = Join-Path $JobDirectory 'return.zip'
if ($RemoveArchive) {
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive }
    return
}
$state = [IO.File]::ReadAllText((Join-Path $JobDirectory 'worker-state.json')) | ConvertFrom-Json
if ($state.status -eq 'running' -or (Get-Process -Id $state.owner_pid -ErrorAction SilentlyContinue)) {
    throw 'Worker still active; export only terminal raw evidence.'
}
$files = @(Get-ChildItem -LiteralPath $JobDirectory -File | Where-Object {
    $_.Extension -in @('.gjf','.log','.err','.json','.chk','.txt')
})
if (Test-Path -LiteralPath $archive) { throw 'Previous return archive exists; inspect it before replacing.' }
Compress-Archive -LiteralPath $files.FullName -DestinationPath $archive -CompressionLevel Fastest
[ordered]@{archive=$archive;files=$files.Count;bytes=(Get-Item -LiteralPath $archive).Length} | ConvertTo-Json
