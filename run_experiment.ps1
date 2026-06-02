param(
    [switch]$Full,
    [string]$ResultsDir = "results",
    [int]$BatchSize = 4,
    [int]$Epochs = 0,
    [string]$Device = "",
    [string]$VideoRoot = "",
    [string]$VideoShards = "",
    [switch]$AllVideoShards,
    [switch]$RequireRealVideos,
    [switch]$KeepShardZip,
    [switch]$CleanupExtractedShards,
    [string]$ResumeCheckpoint = ""
)

$ErrorActionPreference = "Stop"
$mode = if ($Full) { "--full" } else { "--smoke" }
$deviceArgs = @()
if ($Device -ne "") {
    $deviceArgs = @("--device", $Device)
}
$videoArgs = @()
if ($VideoRoot -ne "") {
    $videoArgs = @("--video-root", $VideoRoot)
}
$epochArgs = @()
if ($Epochs -gt 0) {
    $epochArgs = @("--epochs", $Epochs)
}
$shardArgs = @()
if ($VideoShards -ne "") {
    $shardArgs = @("--video-shards", $VideoShards)
}
if ($AllVideoShards) {
    $shardArgs += "--all-video-shards"
}
if ($RequireRealVideos) {
    $videoArgs += "--require-real-videos"
}
if ($KeepShardZip) {
    $shardArgs += "--keep-shard-zip"
}
if ($CleanupExtractedShards) {
    $shardArgs += "--cleanup-extracted-shards"
}
$resumeArgs = @()
if ($ResumeCheckpoint -ne "") {
    $resumeArgs = @("--resume-checkpoint", $ResumeCheckpoint)
}

python experiment.py $mode --results-dir $ResultsDir --batch-size $BatchSize @epochArgs @deviceArgs @videoArgs @shardArgs @resumeArgs
