# ============================================================================
# run.ps1 — Process all CCTV clips and emit events (PowerShell)
# ============================================================================
#
# Usage:
#   .\run.ps1 [-InputDir C:\path\to\clips] [-OutputDir C:\path\to\output]
#             [-ApiUrl http://localhost:8000] [-Synthetic 500]
#
# This script:
#   1. Discovers all video files in the input directory
#   2. Maps filenames to store_id and camera_id based on naming convention
#   3. Runs detect.py for each clip
#   4. Merges all output JSONL files into a single events file
#   5. Optionally ingests into the API
# ============================================================================

param(
    [string]$InputDir = "",
    [string]$OutputDir = "",
    [string]$ApiUrl = "",
    [int]$Synthetic = 0,
    [string]$StoreLayout = "",
    [float]$Fps = 5.0,
    [string]$Detector = "dummy"
)

# --------------------------------------------------------------------------- #
#  Resolve paths                                                              #
# --------------------------------------------------------------------------- #
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

if (-not $InputDir) { $InputDir = Join-Path $ProjectDir "data\clips" }
if (-not $OutputDir) { $OutputDir = Join-Path $ProjectDir "output" }
if (-not $StoreLayout) { $StoreLayout = Join-Path $ProjectDir "data\store_layout.json" }

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$MergedOutput = Join-Path $OutputDir "all_events.jsonl"
# Clear/create merged output
"" | Out-File -FilePath $MergedOutput -Encoding utf8 -NoNewline

# --------------------------------------------------------------------------- #
#  Banner                                                                     #
# --------------------------------------------------------------------------- #
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Store Intelligence - Detection Pipeline"     -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Input:    $InputDir"
Write-Host "  Output:   $OutputDir"
Write-Host "  Detector: $Detector"
Write-Host "  FPS:      $Fps"
Write-Host "  API:      $(if ($ApiUrl) { $ApiUrl } else { 'disabled' })"
Write-Host "==============================================" -ForegroundColor Cyan

# --------------------------------------------------------------------------- #
#  Discover and process video files                                           #
# --------------------------------------------------------------------------- #
$VideoExts = @("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv")
$FoundVideos = 0

if (Test-Path $InputDir) {
    $videos = @()
    foreach ($ext in $VideoExts) {
        $videos += Get-ChildItem -Path $InputDir -Filter $ext -File -ErrorAction SilentlyContinue
    }

    foreach ($video in $videos) {
        $FoundVideos++
        $filename = $video.Name
        Write-Host ""
        Write-Host ">>> Processing: $filename" -ForegroundColor Yellow

        # Parse store_id and camera_id from filename
        # Convention: {STORE_ID}_{CAM_...}_{rest}.ext
        $StoreId = "STORE_BLR_002"
        $CameraId = "CAM_ENTRY_01"

        if ($filename -match "^(STORE_[A-Z]+_\d+)_(CAM_[A-Z]+_\d+)") {
            $StoreId = $Matches[1]
            $CameraId = $Matches[2]
        } else {
            Write-Host "    (Could not parse store/camera from filename, using defaults)" -ForegroundColor DarkGray
        }

        $OutputFile = Join-Path $OutputDir "$($video.BaseName)_events.jsonl"

        Write-Host "    Store:  $StoreId"
        Write-Host "    Camera: $CameraId"
        Write-Host "    Output: $OutputFile"

        # Build and run command
        $cmdArgs = @(
            "-m", "pipeline.detect",
            "--input", $video.FullName,
            "--store", $StoreId,
            "--camera", $CameraId,
            "--output", $OutputFile,
            "--detector", $Detector,
            "--fps", $Fps,
            "--store-layout", $StoreLayout
        )

        if ($ApiUrl) {
            $cmdArgs += @("--api-url", $ApiUrl)
        }

        & python $cmdArgs

        # Append to merged output
        if (Test-Path $OutputFile) {
            Get-Content $OutputFile | Add-Content $MergedOutput
            $lines = (Get-Content $OutputFile | Measure-Object -Line).Lines
            Write-Host "    ✓ Emitted $lines events" -ForegroundColor Green
        }
    }
}

# --------------------------------------------------------------------------- #
#  Synthetic fallback                                                         #
# --------------------------------------------------------------------------- #
if ($FoundVideos -eq 0) {
    if ($Synthetic -eq 0) { $Synthetic = 300 }

    Write-Host ""
    Write-Host ">>> No video files found - running synthetic mode ($Synthetic frames)" -ForegroundColor Yellow

    $OutputFile = Join-Path $OutputDir "synthetic_events.jsonl"

    $cmdArgs = @(
        "-m", "pipeline.detect",
        "--synthetic", $Synthetic,
        "--store", "STORE_BLR_002",
        "--camera", "CAM_ENTRY_01",
        "--output", $OutputFile,
        "--detector", $Detector,
        "--fps", $Fps,
        "--store-layout", $StoreLayout
    )

    if ($ApiUrl) {
        $cmdArgs += @("--api-url", $ApiUrl)
    }

    & python $cmdArgs

    if (Test-Path $OutputFile) {
        Get-Content $OutputFile | Add-Content $MergedOutput
        $lines = (Get-Content $OutputFile | Measure-Object -Line).Lines
        Write-Host "    ✓ Emitted $lines events" -ForegroundColor Green
    }
}

# --------------------------------------------------------------------------- #
#  Summary                                                                    #
# --------------------------------------------------------------------------- #
$totalLines = 0
if (Test-Path $MergedOutput) {
    $totalLines = (Get-Content $MergedOutput | Where-Object { $_.Trim() -ne "" } | Measure-Object -Line).Lines
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Pipeline Complete"                            -ForegroundColor Cyan
Write-Host "  Total events: $totalLines"
Write-Host "  Merged output: $MergedOutput"
Write-Host "==============================================" -ForegroundColor Cyan

if ($ApiUrl -and $totalLines -gt 0) {
    Write-Host "  Events were POSTed to $ApiUrl during processing"
}
