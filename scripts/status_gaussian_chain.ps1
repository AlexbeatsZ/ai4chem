param([Parameter(Mandatory=$true)][string]$JobDirectory)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$JobDirectory = (Resolve-Path -LiteralPath $JobDirectory).Path
if (-not (Test-Path -LiteralPath (Join-Path $JobDirectory 'worker-state.json'))) {
    $task = [IO.File]::ReadAllText((Join-Path $JobDirectory 'task.json')) | ConvertFrom-Json
    [ordered]@{state=@{job_id=$task.job_id;status='queued';raw_chain_completed=$false};
        owner_alive=$false;observed_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Depth 5
    return
}
$state = [IO.File]::ReadAllText((Join-Path $JobDirectory 'worker-state.json')) | ConvertFrom-Json
$logs = @(Get-ChildItem -LiteralPath $JobDirectory -Filter '*.log' | ForEach-Object {
    [ordered]@{name=$_.Name;length=$_.Length;last_write_utc=$_.LastWriteTimeUtc.ToString('o')}
})
$tail = @()
$current = Join-Path $JobDirectory ($state.stage+'.log')
if (Test-Path -LiteralPath $current) {
    # Get-Content -Tail spun on an actively growing 16W log in this environment.
    # Read one bounded byte snapshot; do not chase a moving EOF.
    $share = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $stream = [IO.File]::Open($current,[IO.FileMode]::Open,[IO.FileAccess]::Read,$share)
    try {
        $count = [int][math]::Min(8192,$stream.Length)
        $stream.Seek(-$count,[IO.SeekOrigin]::End) | Out-Null
        $buffer = New-Object byte[] $count
        $readCount = $stream.Read($buffer,0,$count)
        $text = [Text.Encoding]::ASCII.GetString($buffer,0,$readCount)
        $tail = @($text -split "`r?`n" | Select-Object -Last 10)
    } finally { $stream.Dispose() }
}
$processes=@(Get-CimInstance Win32_Process)
[ordered]@{observed_at_utc=[DateTime]::UtcNow.ToString('o');state=$state;
    owner_alive=[bool](@($processes | Where-Object {$_.ProcessId -eq $state.owner_pid}).Count);
    gaussian_processes=@($processes | Where-Object {$_.Name -match '^(g16|l[0-9]+)\.exe$'} |
        Select-Object ProcessId,Name);logs=$logs;current_log_tail=$tail} | ConvertTo-Json -Depth 16
