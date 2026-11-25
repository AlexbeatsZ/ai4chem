# Windows OpenSSH can stall when its output handle is a pipe. Use native CMD
# file redirection, then read the receipt only after the transport exits.
function Invoke-WorkerCommand {
    param(
        [Parameter(Mandatory=$true)][string]$Executable,
        [Parameter(Mandatory=$true)][string[]]$Arguments,
        [int]$TimeoutMilliseconds=180000
    )
    # Only fixed switches and validated space-free worker identities/paths reach
    # this shell command. Reject shell metacharacters rather than reinterpret them.
    if($Executable -notmatch '^[A-Za-z0-9_./:\\ -]+$' -or
        @($Arguments | Where-Object {$_ -notmatch '^[A-Za-z0-9_./:\\@=+-]+$'}).Count) {
        throw 'Unsupported worker command path or argument.'
    }
    if($TimeoutMilliseconds -le 0) { throw 'A finite command timeout is required.' }
    $directory=Join-Path ([IO.Path]::GetTempPath()) ('.agents\ai4chem-transport-'+[Guid]::NewGuid().ToString('N'))
    if($directory -match '[%"\r\n]') { throw 'Unsupported temporary path.' }
    New-Item -ItemType Directory -Path $directory -ErrorAction Stop | Out-Null
    $stdout=Join-Path $directory 'stdout.txt'
    $stderr=Join-Path $directory 'stderr.txt'
    $native='"'+$Executable+'" '+($Arguments -join ' ')+' >"'+$stdout+'" 2>"'+$stderr+'"'
    $process=[Diagnostics.Process]::new()
    $process.StartInfo.FileName=$env:ComSpec
    $process.StartInfo.Arguments='/d /s /c "'+$native+'"'
    $process.StartInfo.UseShellExecute=$false
    $process.StartInfo.CreateNoWindow=$true
    try {
        if(-not $process.Start()) { throw 'Worker transport failed to start.' }
        if(-not $process.WaitForExit($TimeoutMilliseconds)) {
            # This tree consists only of our CMD/native transport, never a
            # scientific owner. End it before removing its temporary output files.
            & (Join-Path $env:WINDIR 'System32\taskkill.exe') /PID $process.Id /T /F | Out-Null
            [void]$process.WaitForExit(5000)
            throw 'Worker transport exceeded its finite timeout; no calculation was restarted.'
        }
        $output=[IO.File]::ReadAllText($stdout)
        $errorText=[IO.File]::ReadAllText($stderr)
        if($process.ExitCode -ne 0) {
            throw ('Worker command failed ('+$process.ExitCode+'): '+$errorText+$output)
        }
        return $output.TrimStart([char]0xfeff)
    } finally {
        $process.Dispose()
        Remove-Item -LiteralPath $directory -Recurse -Force
    }
}
