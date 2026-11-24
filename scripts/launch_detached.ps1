# Launch a long-running python job detached from the current shell (survives the terminal closing).
# Usage (pass the python arguments as ONE quoted string):
#   powershell -ExecutionPolicy Bypass -File scripts\launch_detached.ps1 -Name cvpr_seed42 `
#       -PythonArgs "scripts/run_all_stages.py --config configs/remote_ir_fast.yaml --frames-root C:/data/dataset_frames --run-dir ../runs/cvpr_seed42 --seed 42"
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$PythonArgs,
    [string]$LogDir = "..\runs\logs"
)
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$logDirFull = Join-Path $repo $LogDir
New-Item -ItemType Directory -Force $logDirFull | Out-Null
$out = Join-Path $logDirFull "$Name.log"
$err = Join-Path $logDirFull "$Name.err.log"
$proc = Start-Process -FilePath "python" -ArgumentList ("-u " + $PythonArgs) -WorkingDirectory $repo `
    -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
"$($proc.Id)" | Out-File -Encoding ascii (Join-Path $logDirFull "$Name.pid")
Write-Output "started $Name pid=$($proc.Id)"
Write-Output "stdout: $out"
Write-Output "stderr: $err (tqdm progress bars go here)"
