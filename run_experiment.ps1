param(
    [switch]$Full,
    [string]$ResultsDir = "results",
    [int]$BatchSize = 4,
    [string]$Device = "",
    [string]$VideoRoot = ""
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

python experiment.py $mode --results-dir $ResultsDir --batch-size $BatchSize @deviceArgs @videoArgs
