$ErrorActionPreference = "Continue"

$projectRoot = $PSScriptRoot
Set-Location $projectRoot

$downloadFolder = Join-Path $projectRoot "uploads"
if (-not (Test-Path $downloadFolder)) {
    New-Item -ItemType Directory -Path $downloadFolder | Out-Null
}

$logFile = Join-Path $projectRoot "watcher.log"
$stateFile = Join-Path $projectRoot "watcher_state.json"

# S3 polling configuration
$bucketName = "ballot-imgs"
$watchPrefix = "uploads/"
$pollSeconds = 5
$pipelineUploadPrefix = "raw-images/"
$pythonExe = "python"
$imgExts = @(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

function Global:Write-Log {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts  $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

function Global:As-Array {
    param($InputObject)

    if ($null -eq $InputObject) { return @() }
    if ($InputObject -is [System.Array]) { return $InputObject }
    return @($InputObject)
}

function Global:New-ProcessedMap {
    return @{}
}

function Global:Get-ProcessedKeys {
    if (-not (Test-Path $stateFile)) {
        return New-ProcessedMap
    }

    try {
        $content = Get-Content -Raw -Path $stateFile
        if ([string]::IsNullOrWhiteSpace($content)) {
            return New-ProcessedMap
        }

        $arr = As-Array ($content | ConvertFrom-Json)
        $map = New-ProcessedMap

        foreach ($k in $arr) {
            $ks = [string]$k
            if (-not [string]::IsNullOrWhiteSpace($ks)) {
                $map[$ks] = $true
            }
        }

        return $map
    }
    catch {
        Write-Log "WARNING: Could not read state file '$stateFile': $($_.Exception.Message). Starting with empty state."
        return New-ProcessedMap
    }
}

function Global:Save-ProcessedKeys {
    param([hashtable]$Keys)

    if ($null -eq $Keys) {
        $Keys = New-ProcessedMap
    }

    try {
        $json = @($Keys.Keys | Sort-Object) | ConvertTo-Json -Depth 3
        Set-Content -Path $stateFile -Value $json -Encoding UTF8
    }
    catch {
        Write-Log "WARNING: Failed to save state file '$stateFile': $($_.Exception.Message)"
    }
}

function Global:List-S3Objects {
    param(
        [Parameter(Mandatory=$true)][string]$Bucket,
        [Parameter(Mandatory=$true)][string]$Prefix
    )

    $py = @"
import json
import sys
import boto3

bucket = sys.argv[1]
prefix = sys.argv[2]
s3 = boto3.client('s3')
token = None
out = []

while True:
    params = {'Bucket': bucket, 'Prefix': prefix, 'MaxKeys': 1000}
    if token:
        params['ContinuationToken'] = token
    resp = s3.list_objects_v2(**params)
    for obj in resp.get('Contents', []):
        out.append({'Key': obj.get('Key', ''), 'Size': obj.get('Size', 0)})
    if not resp.get('IsTruncated'):
        break
    token = resp.get('NextContinuationToken')

print(json.dumps(out))
"@

    $tmpDir = [System.IO.Path]::GetTempPath()
    if ([string]::IsNullOrWhiteSpace($tmpDir)) {
        throw "Unable to resolve temp directory."
    }

    $tmpPy = Join-Path $tmpDir "list_s3_objects_watch_downloads.py"
    Set-Content -Path $tmpPy -Value $py -Encoding UTF8

    try {
        $json = & $pythonExe $tmpPy $Bucket $Prefix
        if ($LASTEXITCODE -ne 0) {
            throw "Python list command failed with exit code $LASTEXITCODE"
        }

        if ([string]::IsNullOrWhiteSpace($json)) {
            return @()
        }

        return As-Array ($json | ConvertFrom-Json)
    }
    finally {
        Remove-Item -LiteralPath $tmpPy -ErrorAction SilentlyContinue
    }
}

function Global:Download-S3Object {
    param(
        [Parameter(Mandatory=$true)][string]$Bucket,
        [Parameter(Mandatory=$true)][string]$Key,
        [Parameter(Mandatory=$true)][string]$LocalPath
    )

    $py = @"
import sys
import boto3

bucket = sys.argv[1]
key = sys.argv[2]
dest = sys.argv[3]
s3 = boto3.client('s3')
s3.download_file(bucket, key, dest)
"@

    $tmpDir = [System.IO.Path]::GetTempPath()
    if ([string]::IsNullOrWhiteSpace($tmpDir)) {
        throw "Unable to resolve temp directory."
    }

    $tmpPy = Join-Path $tmpDir "download_s3_object_watch_downloads.py"
    Set-Content -Path $tmpPy -Value $py -Encoding UTF8

    try {
        & $pythonExe $tmpPy $Bucket $Key $LocalPath | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Python download command failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Remove-Item -LiteralPath $tmpPy -ErrorAction SilentlyContinue
    }
}

function Global:Invoke-Pipeline {
    param([Parameter(Mandatory=$true)][string]$ImagePath)

    $args = @(
        "run_full_extraction.py",
        "--images", $ImagePath,
        "--bucket", $bucketName,
        "--s3_prefix", $pipelineUploadPrefix
    )

    Write-Log "Command: $pythonExe $($args -join ' ')"
    $p = Start-Process -FilePath $pythonExe -ArgumentList $args -NoNewWindow -Wait -PassThru
    return $p.ExitCode
}

function Global:Get-S3ObjectKey {
    param($Obj)

    if ($null -eq $Obj) { return "" }

    try {
        $k = $Obj.Key
    }
    catch {
        $k = ""
    }

    return [string]$k
}

Write-Log "Starting S3 watcher in $projectRoot"
Write-Log "Watching s3://$bucketName/$watchPrefix for new images"
Write-Log "Downloaded images folder: $downloadFolder"
Write-Log "Using python: $pythonExe"

$stateExists = Test-Path $stateFile
$processedKeys = Get-ProcessedKeys
if ($null -eq $processedKeys -or $processedKeys -isnot [hashtable]) {
    $processedKeys = New-ProcessedMap
}
Write-Log "Loaded $($processedKeys.Count) processed S3 keys from state."

if (-not $stateExists) {
    try {
        $existingObjects = As-Array (List-S3Objects -Bucket $bucketName -Prefix $watchPrefix)
        $seeded = 0
        foreach ($obj in $existingObjects) {
            $key = Get-S3ObjectKey -Obj $obj
            if ([string]::IsNullOrWhiteSpace($key) -or $key -like '*/') { continue }

            $ext = [string]([System.IO.Path]::GetExtension($key))
            if (-not [string]::IsNullOrWhiteSpace($ext)) { $ext = $ext.ToLowerInvariant() }

            if ($imgExts -contains $ext) {
                if (-not $processedKeys.ContainsKey($key)) {
                    $processedKeys[$key] = $true
                    $seeded++
                }
            }
        }

        Save-ProcessedKeys -Keys $processedKeys
        Write-Log "State file did not exist; seeded $seeded existing S3 images as already processed."
    }
    catch {
        Write-Log "WARNING: Failed initial S3 state seed: $($_.Exception.Message)"
        if ($_.InvocationInfo -and $_.InvocationInfo.PositionMessage) {
            Write-Log "WARNING position: $($_.InvocationInfo.PositionMessage)"
        }
    }
}

while ($true) {
    try {
        $objects = As-Array (List-S3Objects -Bucket $bucketName -Prefix $watchPrefix)
        foreach ($obj in $objects) {
            $key = Get-S3ObjectKey -Obj $obj
            if ([string]::IsNullOrWhiteSpace($key) -or $key -like '*/') { continue }

            $ext = [string]([System.IO.Path]::GetExtension($key))
            if (-not [string]::IsNullOrWhiteSpace($ext)) { $ext = $ext.ToLowerInvariant() }
            if ($imgExts -notcontains $ext) { continue }
            if ($processedKeys.ContainsKey($key)) { continue }

            $safeName = [System.IO.Path]::GetFileName($key)
            if ([string]::IsNullOrWhiteSpace($safeName)) {
                $safeName = [Guid]::NewGuid().ToString() + $ext
            }

            $localPath = Join-Path $downloadFolder $safeName
            if (Test-Path $localPath) {
                $base = [System.IO.Path]::GetFileNameWithoutExtension($safeName)
                $localPath = Join-Path $downloadFolder ("{0}_{1}{2}" -f $base, (Get-Date -Format "yyyyMMddHHmmss"), $ext)
            }

            Write-Log "New S3 image detected: s3://$bucketName/$key"
            Write-Log "Downloading to: $localPath"
            Download-S3Object -Bucket $bucketName -Key $key -LocalPath $localPath

            Write-Log "Running full extraction pipeline for: $localPath"
            $exitCode = Invoke-Pipeline -ImagePath $localPath
            Write-Log "Pipeline finished (ExitCode=$exitCode) for s3://$bucketName/$key"

            if ($exitCode -eq 0) {
                $processedKeys[$key] = $true
                Save-ProcessedKeys -Keys $processedKeys
                Write-Log "Marked as processed (S3 object left unchanged): s3://$bucketName/$key"
            }
            else {
                Write-Log "Extraction failed; will retry on next poll: s3://$bucketName/$key"
            }
        }
    }
    catch {
        Write-Log "ERROR in watcher loop: $($_.Exception.Message)"
        if ($_.InvocationInfo -and $_.InvocationInfo.PositionMessage) {
            Write-Log "ERROR position: $($_.InvocationInfo.PositionMessage)"
        }
    }

    Start-Sleep -Seconds $pollSeconds
}
