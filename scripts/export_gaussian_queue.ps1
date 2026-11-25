param([Parameter(Mandatory=$true)][string]$QueueDirectory)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$QueueDirectory=(Resolve-Path -LiteralPath $QueueDirectory).Path
$state=[IO.File]::ReadAllText((Join-Path $QueueDirectory 'queue-state.json')) | ConvertFrom-Json
$continuationFile=Join-Path $QueueDirectory 'continuation-start.json'
if(Test-Path -LiteralPath $continuationFile){
    $continuation=[IO.File]::ReadAllText($continuationFile)|ConvertFrom-Json
    if($continuation.owner_pid -and (Get-CimInstance Win32_Process -Filter ('ProcessId='+$continuation.owner_pid))){throw 'Continuation still owns or awaits the queue; defer terminal export.'}
}
if($state.status -eq 'running' -or (Get-Process -Id $state.owner_pid -ErrorAction SilentlyContinue)){throw 'Queue still active.'}
foreach($child in @($state.active)){
    if($child.owner_pid -and (Get-CimInstance Win32_Process -Filter ('ProcessId='+$child.owner_pid))){throw 'Scientific child still active; retain partial evidence without exporting.'}
}
$archive=Join-Path $QueueDirectory 'return.zip'
$files=@(Get-ChildItem -LiteralPath $QueueDirectory -Recurse -File | Where-Object {$_.Extension -in @('.gjf','.log','.err','.json','.txt')})
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
function EntryName($file){return $file.FullName.Substring($QueueDirectory.Length+1).Replace('\','/')}
if(Test-Path -LiteralPath $archive){
    $bundle=[IO.Compression.ZipFile]::OpenRead($archive)
    try{
        if($bundle.Entries.Count -ne $files.Count){throw 'Previous terminal archive inventory differs.'}
        foreach($file in $files){
            $entry=$bundle.GetEntry((EntryName $file))
            if(-not $entry -or $entry.Length -ne $file.Length){throw 'Previous terminal archive entry differs.'}
            if($file.Extension -eq '.json'){
                $reader=[IO.StreamReader]::new($entry.Open())
                try{$value=$reader.ReadToEnd()}finally{$reader.Dispose()}
                if($value -cne [IO.File]::ReadAllText($file.FullName)){throw 'Previous terminal archive identity differs.'}
            }
        }
    }finally{$bundle.Dispose()}
}else{
    $partial=Join-Path $QueueDirectory 'return.partial.zip'
    $bundle=[IO.Compression.ZipFile]::Open($partial,[IO.Compression.ZipArchiveMode]::Create)
    try{
        foreach($file in $files){[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($bundle,$file.FullName,(EntryName $file),[IO.Compression.CompressionLevel]::Fastest)|Out-Null}
    }finally{$bundle.Dispose()}
    Move-Item -LiteralPath $partial -Destination $archive
}
[ordered]@{archive=$archive;files=$files.Count;bytes=(Get-Item -LiteralPath $archive).Length;status=$state.status;checkpoints_excluded=$true}|ConvertTo-Json
