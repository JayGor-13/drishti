param(
    [switch]$Full,
    [string]$ResultsDir = "results",
    [int]$BatchSize = 4,
    [int]$Epochs = 0,
    [string]$Device = "",
    [string]$MetadataFile = "",
    [string]$VideoRoot = "",
    [switch]$RequireRealVideos,
    [string]$ResumeCheckpoint = ""
)

$ErrorActionPreference = "Stop"
$mode = if ($Full) { "--full" } else { "--smoke" }
$deviceArgs = @()
if ($Device -ne "") {
    $deviceArgs = @("--device", $Device)
}
$videoArgs = @()
if ($MetadataFile -ne "") {
    $videoArgs += @("--metadata-file", $MetadataFile)
}
if ($VideoRoot -ne "") {
    $videoArgs += @("--video-root", $VideoRoot)
}
$epochArgs = @()
if ($Epochs -gt 0) {
    $epochArgs = @("--epochs", $Epochs)
}
if ($RequireRealVideos) {
    $videoArgs += "--require-real-videos"
}
$resumeArgs = @()
if ($ResumeCheckpoint -ne "") {
    $resumeArgs = @("--resume-checkpoint", $ResumeCheckpoint)
}

python experiment.py $mode --results-dir $ResultsDir --batch-size $BatchSize @epochArgs @deviceArgs @videoArgs @resumeArgs
