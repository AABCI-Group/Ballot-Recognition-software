$ErrorActionPreference = "Continue"

# Assume this script sits in your project root
$projectRoot = $PSScriptRoot
Set-Location $projectRoot

# Folder to watch -> "uploads" folder inside the project
$folder = Join-Path $projectRoot "uploads"

if (-not (Test-Path $folder)) {
    Write-Host "Uploads folder not found at: $folder"
    Write-Host "Creating uploads folder..."
    New-Item -ItemType Directory -Path $folder | Out-Null
}

# Log file
$logFile = Join-Path $projectRoot "watcher.log"

function Global:Write-Log {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts  $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

# Global variables so event handlers can see them
$global:pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$global:weights   = Join-Path $projectRoot "runs\train\yolo_stamp\weights\best.pt"
$global:outDir    = Join-Path $projectRoot "outputs"

Write-Log "Starting watcher in $projectRoot, watching folder: $folder"
Write-Log "Using python: $global:pythonExe"

# Set up the file system watcher
$fsw = New-Object System.IO.FileSystemWatcher
$fsw.Path = $folder
$fsw.IncludeSubdirectories = $false
$fsw.Filter = "*.*"
$fsw.NotifyFilter = [IO.NotifyFilters]'FileName, LastWrite, CreationTime'
$fsw.EnableRaisingEvents = $true

# CREATED handler: run the model
Register-ObjectEvent -InputObject $fsw -EventName Created -Action {
    $path = $Event.SourceEventArgs.FullPath
    $ext  = [System.IO.Path]::GetExtension($path).ToLower()

    Write-Log "Created event: $path (ext=$ext)"

    # Extensions we care about
    $imgExts = ".png", ".jpg", ".jpeg", ".tif", ".tiff"

    if ($imgExts -contains $ext) {
        Write-Log "New image detected: $path"
        Write-Log "About to run python for: $path"

        # Log exact command
        $cmdLine = "`"$global:pythonExe`" -m src.infer.predict --weights `"$global:weights`" --images `"$path`" --out_dir `"$global:outDir`""
        Write-Log "Command: $cmdLine"

        try {
            & $global:pythonExe -m src.infer.predict `
                --weights $global:weights `
                --images  $path `
                --out_dir $global:outDir 2>&1 |
                ForEach-Object { Write-Log "python: $_" }

            Write-Log "Python finished for $path with exit code $LASTEXITCODE"
        }
        catch {
            Write-Log "ERROR running python: $($_.Exception.Message)"
        }
    }
} | Out-Null

# CHANGED handler – just logs
Register-ObjectEvent -InputObject $fsw -EventName Changed -Action {
    $path = $Event.SourceEventArgs.FullPath
    $ext  = [System.IO.Path]::GetExtension($path).ToLower()
    Write-Log "Changed event: $path (ext=$ext)"
} | Out-Null

# RENAMED handler – just logs
Register-ObjectEvent -InputObject $fsw -EventName Renamed -Action {
    $path = $Event.SourceEventArgs.FullPath
    $ext  = [System.IO.Path]::GetExtension($path).ToLower()
    Write-Log "Renamed event: $path (ext=$ext)"
} | Out-Null

Write-Log "Watching $folder for new image files... Press Ctrl+C to stop."

# Keep this PowerShell process alive
while ($true) {
    Wait-Event -Timeout 5 | Out-Null
}
