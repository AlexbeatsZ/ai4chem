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
if (Test-Path -LiteralPath $archive) {
    # A disconnected export may have completed compression before its receipt
    # reached the caller. Reuse only an archive matching the terminal evidence.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $bundle = [IO.Compression.ZipFile]::OpenRead($archive)
    try {
        if($bundle.Entries.Count -ne $files.Count) { throw 'Previous archive inventory differs; preserve it for diagnosis.' }
        foreach($file in $files) {
            $entries = @($bundle.Entries | Where-Object { $_.FullName -ceq $file.Name })
            if($entries.Count -ne 1 -or $entries[0].Length -ne $file.Length) {
                throw ('Previous archive entry differs: '+$file.Name)
            }
            if($file.Extension -eq '.json') {
                $reader = [IO.StreamReader]::new($entries[0].Open())
                try { $archivedText = $reader.ReadToEnd() } finally { $reader.Dispose() }
                if($archivedText -cne [IO.File]::ReadAllText($file.FullName)) {
                    throw ('Previous archive identity differs: '+$file.Name)
                }
            }
        }
    } finally { $bundle.Dispose() }
} else {
    Compress-Archive -LiteralPath $files.FullName -DestinationPath $archive -CompressionLevel Fastest
}
[ordered]@{archive=$archive;files=$files.Count;bytes=(Get-Item -LiteralPath $archive).Length} | ConvertTo-Json
