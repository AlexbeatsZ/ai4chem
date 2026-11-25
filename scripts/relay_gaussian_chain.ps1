param(
    [Parameter(Mandatory=$true)][ValidateSet('Status','Export','Cleanup')][string]$Action,
    [Parameter(Mandatory=$true)][string]$WorkerRoot,
    [Parameter(Mandatory=$true)][string]$JobDirectory,
    [string]$WorkerSsh = 'ai4chem-worker'
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
. (Join-Path $PSScriptRoot 'worker_transport.ps1')
foreach($value in @($WorkerRoot,$JobDirectory,$WorkerSsh)) {
    if($value -notmatch '^[A-Za-z0-9_./:\\@-]+$') { throw 'Use simple space-free worker paths and SSH identity.' }
}
function Remote([string]$Script,[switch]$RemoveArchive,[switch]$StreamOutput) {
    $arguments = @('-n','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',$WorkerSsh,
        'powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',
        ($WorkerRoot+'/scripts/'+$Script),'-JobDirectory',$JobDirectory)
    if($RemoveArchive) { $arguments += '-RemoveArchive' }
    if($StreamOutput) {
        & ssh @arguments
        if($LASTEXITCODE -ne 0) { throw "Worker command failed: $Script" }
        return
    }
    return (Invoke-WorkerCommand -Executable (Get-Command ssh -CommandType Application).Source -Arguments $arguments)
}
if($Action -eq 'Status') { Remote 'status_gaussian_chain.ps1' -StreamOutput; return }
if($Action -eq 'Cleanup') { Remote 'export_gaussian_chain.ps1' -RemoveArchive; return }
$project = Split-Path -Parent $PSScriptRoot
$jobName = Split-Path -Leaf $JobDirectory
$directory = Join-Path $project ('results\coordination\diversity-dft\relay\'+$jobName)
New-Item -ItemType Directory -Path $directory -Force | Out-Null
$receipt = Join-Path $directory 'relay-export.json'
if(Test-Path -LiteralPath $receipt) {
    $exported = [IO.File]::ReadAllText($receipt) | ConvertFrom-Json
} else {
    $raw = Remote 'export_gaussian_chain.ps1'
    $exported = $raw | ConvertFrom-Json
    [IO.File]::WriteAllText($receipt,$raw,[Text.UTF8Encoding]::new($false))
}
$archive = Join-Path $directory 'return.zip'
if(-not (Test-Path -LiteralPath $archive)) {
    $partial = Join-Path $directory 'return.partial.zip'
    $source = $WorkerSsh+':'+$exported.archive.Replace('\','/')
    & scp -q -o BatchMode=yes -o ConnectTimeout=10 $source $partial
    if($LASTEXITCODE -ne 0) { throw 'Worker evidence relay failed; receipt retained for retry.' }
    Move-Item -LiteralPath $partial -Destination $archive
}
if((Get-Item -LiteralPath $archive).Length -ne $exported.bytes) { throw 'Relayed archive size differs from export receipt.' }
[ordered]@{archive=$archive;bytes=$exported.bytes;files=$exported.files;job=$jobName} | ConvertTo-Json
