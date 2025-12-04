$ErrorActionPreference = "Continue" 

$projectRoot = $PSScriptRoot
Set-Location $projectRoot

$folder = Join-Path $projectRoot "uploads"

if (-not (Test-Path $folder)) {
    Write-Host "Uploads folder not found at: $folder"
    Write-Host "Creating uploads folder..."
    New-Item -ItemType Directory -Path $folder | Out-Null
}

$logFile = Join-Path $projectRoot "watcher.log"

function Global:Write-Log {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts  $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

# ✅ Just use the literal exe name (or full path if you want)
$pythonExe = "python"

Write-Log "Starting watcher in $projectRoot, watching folder: $folder"
Write-Log "Using python: $pythonExe"

$fsw = New-Object System.IO.FileSystemWatcher
$fsw.Path = $folder
$fsw.IncludeSubdirectories = $false
$fsw.Filter = "*.*"
$fsw.NotifyFilter = [IO.NotifyFilters]'FileName, LastWrite, CreationTime'
$fsw.EnableRaisingEvents = $true

Register-ObjectEvent -InputObject $fsw -EventName Created -Action {
    $path = $Event.SourceEventArgs.FullPath
    $ext  = [System.IO.Path]::GetExtension($path).ToLower()

    Write-Log "Created event: $path (ext=$ext)"

    $imgExts = ".png", ".jpg", ".jpeg", ".tif", ".tiff"

    if ($imgExts -contains $ext) {
        Write-Log "New image detected: $path"
        Write-Log "Running full extraction pipeline..."

        # NOTE: uploads/ is relative to $projectRoot (where you started the script)
        $imagesArg = "uploads/"

        # Build arguments as an *array* (no quoting headaches)
        $args = @(
            "-m", "src.runtime.run_full_extraction",
            "--images", $imagesArg,
            "--bucket", "ballot-imgs",
            "--s3_prefix", "raw-images/"
        )

        # Hard-code python name here to avoid scope madness
        $py = "python"

        Write-Log "Command: $py $($args -join ' ')"

        try {
            Write-Log "Starting python process..."
            Start-Process -FilePath $py -ArgumentList $args -NoNewWindow -Wait
            Write-Log "Python full extraction finished for $path"
        }
        catch {
            Write-Log "ERROR running python: $($_.Exception.Message)"
        }
    }
} | Out-Null

Register-ObjectEvent -InputObject $fsw -EventName Changed -Action {
    $path = $Event.SourceEventArgs.FullPath
    $ext  = [System.IO.Path]::GetExtension($path).ToLower()
    Write-Log "Changed event: $path (ext=$ext)"
} | Out-Null

Register-ObjectEvent -InputObject $fsw -EventName Renamed -Action {
    $path = $Event.SourceEventArgs.FullPath
    $ext  = [System.IO.Path]::GetExtension($path).ToLower()
    Write-Log "Renamed event: $path (ext=$ext)"
} | Out-Null

Write-Log "Watching $folder for new image files... Press Ctrl+C to stop."

while ($true) {
    Wait-Event -Timeout 5 | Out-Null
}
