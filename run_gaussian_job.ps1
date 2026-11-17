param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [string]$GaussianDirectory = (Join-Path $env:LOCALAPPDATA 'Programs\g16w')
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$inputPath = (Resolve-Path -LiteralPath $InputFile).Path
$folder = Split-Path -Parent $inputPath
$inputName = Split-Path -Leaf $inputPath
$outputName = [IO.Path]::ChangeExtension($inputName, '.log')
$outputPath = Join-Path $folder $outputName
$env:GAUSS_EXEDIR = $GaussianDirectory
$env:GAUSS_SCRDIR = $folder
$process = Start-Process -FilePath (Join-Path $GaussianDirectory 'g16.exe') `
    -ArgumentList @($inputName, $outputName) -WorkingDirectory $folder `
    -RedirectStandardOutput (Join-Path $folder ($inputName + '.launcher.log')) `
    -RedirectStandardError (Join-Path $folder ($inputName + '.launcher.err')) `
    -WindowStyle Hidden -Wait -PassThru
if (-not (Test-Path -LiteralPath $outputPath)) {
    throw "Gaussian produced no output at $outputPath"
}
$logText = [IO.File]::ReadAllText($outputPath)
if ($process.ExitCode -ne 0 -or $logText.TrimEnd() -notmatch 'Normal termination of Gaussian[^\r\n]*\z') {
    throw "Gaussian failed. Inspect $outputPath"
}
Write-Output "Gaussian completed $outputPath"
