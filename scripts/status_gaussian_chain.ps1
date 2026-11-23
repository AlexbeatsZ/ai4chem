param([Parameter(Mandatory=$true)][string]$JobDirectory)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$JobDirectory = (Resolve-Path -LiteralPath $JobDirectory).Path
$state = Get-Content -LiteralPath (Join-Path $JobDirectory 'worker-state.json') -Raw | ConvertFrom-Json
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
[ordered]@{observed_at_utc=[DateTime]::UtcNow.ToString('o');state=$state;
    owner_alive=[bool](Get-Process -Id $state.owner_pid -ErrorAction SilentlyContinue);
    gaussian_processes=@(Get-Process | Where-Object {$_.ProcessName -match '^(g16|l[0-9]+)$'} |
        Select-Object Id,ProcessName,CPU);logs=$logs;current_log_tail=$tail} | ConvertTo-Json -Depth 16
