# Capture native SSH output outside the Windows PowerShell pipeline. Terminal
# export captures otherwise left orphaned relay processes after client timeouts.
function Invoke-WorkerCommand {
    param(
        [Parameter(Mandatory=$true)][string]$Executable,
        [Parameter(Mandatory=$true)][string[]]$Arguments,
        [int]$TimeoutMilliseconds=180000
    )
    # These transports accept only the already validated, space-free paths and
    # fixed switches. Do not silently reinterpret a joined argument containing spaces.
    if(@($Arguments | Where-Object {$_ -match '\s|"'}).Count) {
        throw 'Worker command arguments must be space-free and unquoted.'
    }
    if($TimeoutMilliseconds -le 0) { throw 'A finite command timeout is required.' }
    $process=[Diagnostics.Process]::new()
    $process.StartInfo.FileName=$Executable
    $process.StartInfo.Arguments=$Arguments -join ' '
    $process.StartInfo.UseShellExecute=$false
    $process.StartInfo.CreateNoWindow=$true
    $process.StartInfo.RedirectStandardOutput=$true
    $process.StartInfo.RedirectStandardError=$true
    $process.StartInfo.StandardOutputEncoding=[Text.UTF8Encoding]::new($false)
    $process.StartInfo.StandardErrorEncoding=[Text.UTF8Encoding]::new($false)
    try {
        if(-not $process.Start()) { throw 'Worker transport failed to start.' }
        # Drain both native streams concurrently before waiting for process exit.
        $stdout=$process.StandardOutput.ReadToEndAsync()
        $stderr=$process.StandardError.ReadToEndAsync()
        if(-not $process.WaitForExit($TimeoutMilliseconds)) {
            $process.Kill()
            $process.WaitForExit()
            throw 'Worker transport exceeded its finite timeout; no calculation was restarted.'
        }
        $process.WaitForExit()
        $output=$stdout.GetAwaiter().GetResult()
        $errorText=$stderr.GetAwaiter().GetResult()
        if($process.ExitCode -ne 0) {
            throw ('Worker command failed ('+$process.ExitCode+'): '+$errorText+$output)
        }
        return $output.TrimStart([char]0xfeff)
    } finally {
        $process.Dispose()
    }
}
