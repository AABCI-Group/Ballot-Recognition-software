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

function Global:Wait-FileReady {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [int]$TimeoutSeconds = 30,
        [int]$PollMs = 250
    )

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        try {
            # If we can open with no sharing, writer is done.
            $fs = [System.IO.File]::Open($Path, 'Open', 'ReadWrite', 'None')
            $fs.Close()
            return $true
        }
        catch {
            Start-Sleep -Milliseconds $PollMs
        }
    }
    return $false
}

function Global:Try-DeleteFile {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [int]$Retries = 10,
        [int]$DelayMs = 300
    )

    for ($i=1; $i -le $Retries; $i++) {
        try {
            if (Test-Path $Path) {
                Remove-Item -LiteralPath $Path -Force -ErrorAction Stop
            }
            return $true
        }
        catch {
            Start-Sleep -Milliseconds $DelayMs
        }
    }
    return $false
}

# Just use the literal exe name (or full path if you want)
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

        # Wait until the file is fully written
        if (-not (Wait-FileReady -Path $path -TimeoutSeconds 60)) {
            Write-Log "ERROR: File not ready after timeout, skipping: $path"
            return
        }

        Write-Log "Running full extraction pipeline..."

        # NOTE: uploads/ is relative to $projectRoot (where you started the script)
        $imagesArg = "uploads/"

        $args = @(
            "run_full_extraction.py",
            "--images", $imagesArg,
            "--bucket", "ballot-imgs",
            "--s3_prefix", "raw-images/"
        )

        $py = "python"

        Write-Log "Command: $py $($args -join ' ')"

        try {
            Write-Log "Starting python process..."
            $p = Start-Process -FilePath $py -ArgumentList $args -NoNewWindow -Wait -PassThru
            Write-Log "Python full extraction finished (ExitCode=$($p.ExitCode)) for $path"

            if ($p.ExitCode -eq 0) {
                Write-Log "Deleting processed image: $path"
                if (Try-DeleteFile -Path $path) {
                    Write-Log "Deleted: $path"
                } else {
                    Write-Log "ERROR: Failed to delete after retries: $path"
                }
            } else {
                Write-Log "Not deleting $path because extraction failed (ExitCode=$($p.ExitCode))"
            }
        }
        catch {
            Write-Log "ERROR running python: $($_.Exception.Message)"
            Write-Log "Not deleting $path due to error."
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
