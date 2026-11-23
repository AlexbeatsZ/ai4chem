param(
    [string]$GaussianDirectory = 'C:\G16W',
    [string]$OutputDirectory = ''
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$root = Split-Path -Parent $PSScriptRoot
if (-not $OutputDirectory) {
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $OutputDirectory = Join-Path $root "results\node-readiness\$env:COMPUTERNAME\$stamp"
}
$executable = Join-Path $GaussianDirectory 'g16.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Gaussian executable not found: $executable"
}
if (Test-Path -LiteralPath (Join-Path $OutputDirectory 'water.gjf')) {
    throw "Use a fresh output directory; existing readiness evidence is preserved."
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$inputFile = Join-Path $OutputDirectory 'water.gjf'
$inputText = @'
%chk=water.chk
%mem=2GB
%nprocshared=4
#p wB97X/6-31G(d) opt freq scf=(xqc,maxcycle=128) int=ultrafine nosymm

ai4chem Gaussian node readiness - isolated water Opt/Freq

0 1
O 0.000000000000 0.000000000000 0.000000000000
H 0.000000000000 0.757000000000 0.586000000000
H 0.000000000000 -0.757000000000 0.586000000000


'@
[IO.File]::WriteAllText($inputFile, $inputText.Replace("`r`n", "`n") + "`n", [Text.Encoding]::ASCII)
$started = [DateTime]::UtcNow
$timer = [Diagnostics.Stopwatch]::StartNew()
$summary = [ordered]@{
    host = $env:COMPUTERNAME
    started_at_utc = $started.ToString('o')
    gaussian_directory = $GaussianDirectory
    method = 'wB97X/6-31G(d)'
    threads = 4
    memory = '2GB'
    purpose = 'Node readiness only; not a v0.5.0 reaction evaluation'
    status = 'running'
}
try {
    & (Join-Path $root 'run_gaussian_job.ps1') -InputFile $inputFile -GaussianDirectory $GaussianDirectory
    $log = [IO.File]::ReadAllText((Join-Path $OutputDirectory 'water.log'))
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $energyMatches = [regex]::Matches($log, 'SCF Done:\s+E\([^\r\n]*?=\s+(-?\d+\.\d+)')
    if ($energyMatches.Count -eq 0) { throw 'No SCF energy in Gaussian output.' }
    $energy = [double]::Parse($energyMatches[$energyMatches.Count - 1].Groups[1].Value, $culture)
    $frequencyLines = [regex]::Matches($log, '(?m)^\s*Frequencies\s+--\s+([^\r\n]+)')
    $frequencies = @($frequencyLines | ForEach-Object {
        [regex]::Matches($_.Groups[1].Value, '-?\d+\.\d+') | ForEach-Object {
            [double]::Parse($_.Value, $culture)
        }
    })
    $optimized = $log -match 'Optimization completed\.'
    if (-not $optimized -or $frequencies.Count -ne 3 -or @($frequencies | Where-Object {$_ -le 0}).Count -ne 0) {
        throw 'Water readiness requires completed optimization and exactly three positive vibrational frequencies.'
    }
    $version = [regex]::Match($log, '(?m)^.*Gaussian 16:.*$').Value.Trim()
    $summary.status = 'passed'
    $summary.energy_hartree = $energy
    $summary.frequencies_cm1 = $frequencies
    $summary.optimization_completed = $optimized
    $summary.gaussian_banner = $version
} catch {
    $summary.status = 'failed'
    $summary.error = $_.Exception.Message
    throw
} finally {
    $timer.Stop()
    $summary.elapsed_seconds = $timer.Elapsed.TotalSeconds
    $summary.finished_at_utc = [DateTime]::UtcNow.ToString('o')
    $summary.output_directory = $OutputDirectory
    $json = $summary | ConvertTo-Json -Depth 5
    [IO.File]::WriteAllText((Join-Path $OutputDirectory 'summary.json'), $json + "`n", [Text.UTF8Encoding]::new($false))
    Write-Output $json
}
