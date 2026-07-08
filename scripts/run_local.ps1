# End-to-end pipeline runner for the Windows CUDA machine.
#
# Prereqs (one-time):
#   - NVIDIA driver installed (check: nvidia-smi)
#   - uv installed:  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#   - repo cloned/synced INCLUDING Data\Parkovani_praha.geojson
#
# Usage (from the repo root):
#   .\scripts\run_local.ps1 -Stage setup     # env + CUDA check
#   .\scripts\run_local.ps1 -Stage prep      # labels + sheet index + alignment QC
#   .\scripts\run_local.ps1 -Stage chips     # full chip build (~8 GB downloads, hours)
#   .\scripts\run_local.ps1 -Stage train     # full training (overnight)
#   .\scripts\run_local.ps1 -Stage train -Resume        # continue interrupted training
#   .\scripts\run_local.ps1 -Stage infer -Checkpoint data\runs\<run>\best.pt
#   .\scripts\run_local.ps1 -Stage all       # setup -> prep -> chips -> train
#
# If training hits CUDA out-of-memory, lower the batch size:
#   .\scripts\run_local.ps1 -Stage train -ExtraArgs @('--train.batch_size','8')

param(
    [ValidateSet('setup', 'prep', 'chips', 'train', 'assess', 'infer', 'all')]
    [string]$Stage = 'all',
    [switch]$Resume,
    [string]$Checkpoint = '',
    [string]$Aoi = '',
    [int]$MaxChips = 100000,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'   # Czech sheet names in console logs

function Invoke-Step([string]$Name, [string[]]$CmdArgs) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & uv run python @CmdArgs @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit $LASTEXITCODE)" }
}

function Do-Setup {
    Write-Host "=== setup: uv sync (CUDA torch on Windows via pyproject markers) ===" -ForegroundColor Cyan
    & nvidia-smi | Select-Object -First 12
    & uv sync
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }
    & uv run python -c @"
import torch
ok = torch.cuda.is_available()
print('torch', torch.__version__, '| CUDA available:', ok)
if ok:
    print('GPU:', torch.cuda.get_device_name(0),
          '| VRAM: %.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    raise SystemExit('CUDA NOT available - check driver / that uv picked +cu128 wheels')
"@
    if ($LASTEXITCODE -ne 0) { throw 'CUDA check failed' }
}

function Do-Prep {
    Invoke-Step 'labels -> EPSG:5514 gpkg' @('-m', 'parking.geo')
    Invoke-Step 'ATOM sheet index' @('-m', 'parking.acquire', 'index')
    Invoke-Step 'alignment QC overlay (inspect data\qc\PRAH74_overlay_zoom.png!)' `
        @('-m', 'parking.qc', 'overlay', '--sheet', 'PRAH74', '--zoom-to-labels')
}

function Do-Chips {
    # streams: download sheet -> cut chips -> delete sheet; safe to re-run (resumes)
    Invoke-Step "chip build, cap $MaxChips" @('-m', 'parking.chips', '--all', '--chips.max_chips', "$MaxChips")
    Invoke-Step 'chip/mask QC mosaic (inspect data\qc\chips_mosaic.png)' @('-m', 'parking.qc', 'chips')
}

function Do-Train {
    $cmd = @('-m', 'parking.train')
    if ($Resume) { $cmd += '--resume' }
    Invoke-Step 'training' $cmd
}

function Do-Assess {
    # visual check of segmentation quality on random val + train chips
    $cmd = @('-m', 'parking.qc', 'predict')
    if ($Checkpoint) { $cmd += @('--checkpoint', $Checkpoint) }
    Invoke-Step 'predictions on val chips   -> data\qc\predict_val.png' ($cmd + @('--split', 'val'))
    Invoke-Step 'predictions on train chips -> data\qc\predict_train.png' ($cmd + @('--split', 'train'))
}

function Do-Infer {
    if (-not $Checkpoint) {
        $Checkpoint = Get-ChildItem data\runs\*\best.pt -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime | Select-Object -Last 1 -ExpandProperty FullName
        if (-not $Checkpoint) { throw 'no best.pt found; pass -Checkpoint' }
        Write-Host "using latest checkpoint: $Checkpoint"
    }
    $cmd = @('-m', 'parking.infer', '--checkpoint', $Checkpoint)
    if ($Aoi) { $cmd += @('--aoi', $Aoi) }
    else {
        # default: the 18 held-out validation sheets around Prague would be listed
        # in the chip index; for a quick look just do one dense sheet
        $cmd += @('--sheets', 'PRAH74')
    }
    Invoke-Step 'inference + per-sheet polygons' $cmd
    Invoke-Step 'merge polygon layers' @('-m', 'parking.polygonize', 'merge')
}

switch ($Stage) {
    'setup' { Do-Setup }
    'prep'  { Do-Prep }
    'chips' { Do-Chips }
    'train'  { Do-Train }
    'assess' { Do-Assess }
    'infer'  { Do-Infer }
    'all'    { Do-Setup; Do-Prep; Do-Chips; Do-Train; Do-Assess }
}
Write-Host "`nDone." -ForegroundColor Green
