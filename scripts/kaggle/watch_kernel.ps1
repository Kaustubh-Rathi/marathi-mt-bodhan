# Watch a Kaggle kernel until it finishes, saving logs locally.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/kaggle/watch_kernel.ps1 -Task bodhan [-Acct 1] [-IntervalSec 20]
param(
  [Parameter(Mandatory = $true)][ValidateSet("prepare","bodhan","indictrans2","eval","ablation","probe","probe2","probe3")][string]$Task,
  [ValidateSet("1","2","3")][string]$Acct = "",
  [int]$IntervalSec = 20,
  [int]$MaxMinutes = 720
)
$ErrorActionPreference = "Stop"

$acctMap = @{ "1" = "$HOME\.kaggle\access_token"; "2" = "$HOME\.kaggle\access_token_acct2"; "3" = "$HOME\.kaggle\access_token_acct3" }
$taskOwner = @{ "prepare"="acajjhfh"; "bodhan"="kaustubhcrathi"; "indictrans2"="dreamexcellence"; "eval"="kaustubhcrathi"; "ablation"="acajjhfh"; "probe"="kaustubhcrathi"; "probe2"="dreamexcellence"; "probe3"="acajjhfh" }
$taskAcct = @{ "prepare"="3"; "bodhan"="1"; "indictrans2"="2"; "eval"="1"; "ablation"="3"; "probe"="1"; "probe2"="2"; "probe3"="3" }
if (-not $Acct) { $Acct = $taskAcct[$Task] }

$metaPath = Join-Path $PSScriptRoot "kernel-metadata.$Task.json"
$meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
$slug = $meta.id
$logDir = Join-Path (Split-Path $PSScriptRoot -Parent | Split-Path -Parent) "artifacts\$slug"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logFile = Join-Path $logDir "kernel_logs.json"

$env:KAGGLE_API_TOKEN = (Get-Content -LiteralPath $acctMap[$Acct] -Raw).Trim()
$env:PYTHONIOENCODING = "utf-8"
Write-Host "watching $slug (acct $Acct); logs -> $logFile"

$deadline = (Get-Date).AddMinutes($MaxMinutes)
while ((Get-Date) -lt $deadline) {
  kaggle kernels logs $slug 2>$null | Out-File -FilePath $logFile -Encoding utf8
  $status = (kaggle kernels status $slug 2>&1 | Out-String)
  $state = if ($status -match "RUNNING") { "RUNNING" } elseif ($status -match "COMPLETE") { "COMPLETE" } elseif ($status -match "ERROR|FAILED|CANCELLED") { "FAILED" } else { "UNKNOWN" }
  $last = (Get-Content $logFile -ErrorAction SilentlyContinue | Select-Object -Last 1)
  Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $state)
  if ($state -ne "RUNNING") { Write-Host "final: $state"; break }
  Start-Sleep -Seconds $IntervalSec
}
Remove-Item Env:\KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
Write-Host "done. parse logs: python parse_logs.py `"$logFile`""
