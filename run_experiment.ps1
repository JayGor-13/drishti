param(
    [switch]$Full,
    [string]$ResultsDir = "results",
    [int]$BatchSize = 2,
    [int]$Epochs = 0,
    [string]$Device = "",
    [string]$TrainImageRoot = "",
    [string]$TrainAnnFile = "",
    [string]$ValImageRoot = "",
    [string]$ValAnnFile = "",
    [string]$Stage = "all",
    [string]$ResumeCheckpoint = ""
)

$ErrorActionPreference = "Stop"
$mode = if ($Full) { "--full" } else { "--smoke" }
$deviceArgs = @()
if ($Device -ne "") {
    $deviceArgs = @("--device", $Device)
}
$dataArgs = @()
if ($TrainImageRoot -ne "") {
    $dataArgs += @("--train-image-root", $TrainImageRoot)
}
if ($TrainAnnFile -ne "") {
    $dataArgs += @("--train-ann-file", $TrainAnnFile)
}
if ($ValImageRoot -ne "") {
    $dataArgs += @("--val-image-root", $ValImageRoot)
}
if ($ValAnnFile -ne "") {
    $dataArgs += @("--val-ann-file", $ValAnnFile)
}
$epochArgs = @()
if ($Epochs -gt 0) {
    $epochArgs = @("--epochs", $Epochs)
}
$resumeArgs = @()
if ($ResumeCheckpoint -ne "") {
    $resumeArgs = @("--resume-checkpoint", $ResumeCheckpoint)
}

python experiment.py $mode --stage $Stage --results-dir $ResultsDir --batch-size $BatchSize @epochArgs @deviceArgs @dataArgs @resumeArgs
