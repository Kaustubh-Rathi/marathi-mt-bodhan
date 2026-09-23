# Push exactly ONE Kaggle script kernel with the right per-account token.
#
# kaggle kernels push -p <dir> requires <dir>/kernel-metadata.json, but this
# repo keeps four named metadata files (kernel-metadata.<task>.json), so this
# helper stages a single kernel into a temp dir and pushes that.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/kaggle/push_kernel.ps1 -Task bodhan
#   powershell -ExecutionPolicy Bypass -File scripts/kaggle/push_kernel.ps1 -Task indictrans2 -Acct 2
#   powershell -ExecutionPolicy Bypass -File scripts/kaggle/push_kernel.ps1 -Task prepare -DryRun
param(
  [Parameter(Mandatory = $true)][ValidateSet("prepare", "bodhan", "indictrans2", "eval")][string]$Task,
  [ValidateSet("1", "2", "3")][string]$Acct = "",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Account -> task mapping (scripts/kaggle/README.md); overridable with -Acct.
if (-not $Acct) {
  $Acct = switch ($Task) {
    "prepare"     { "3" }
    "bodhan"      { "1" }
    "indictrans2" { "2" }
    "eval"        { "1" }
  }
}

$metadata = Join-Path $PSScriptRoot "kernel-metadata.$Task.json"
if (-not (Test-Path -LiteralPath $metadata)) { throw "missing metadata file: $metadata" }
$meta = Get-Content -LiteralPath $metadata -Raw | ConvertFrom-Json

$codeFile = Join-Path $PSScriptRoot $meta.code_file
if (-not (Test-Path -LiteralPath $codeFile)) { throw "missing code_file: $codeFile" }

$staging = Join-Path $env:TEMP "mr-mt-push-$Task"
Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $staging | Out-Null
Copy-Item -LiteralPath $metadata (Join-Path $staging "kernel-metadata.json")
Copy-Item -LiteralPath $codeFile $staging

$tokenFile = switch ($Acct) {
  "1" { Join-Path $HOME ".kaggle\access_token" }
  "2" { Join-Path $HOME ".kaggle\access_token_acct2" }
  "3" { Join-Path $HOME ".kaggle\access_token_acct3" }
}
if (-not (Test-Path -LiteralPath $tokenFile)) { throw "no Kaggle token for acct $Acct : $tokenFile" }

Write-Output "task        : $Task"
Write-Output "kernel id   : $($meta.id)"
Write-Output "account     : acct$Acct  ($tokenFile)"
Write-Output "staging dir : $staging  (kernel-metadata.json + $($meta.code_file))"
Write-Output "gpu/internet: $($meta.enable_gpu) / $($meta.enable_internet)"
Write-Output "datasets    : $($meta.dataset_sources -join ', ')"

if ($DryRun) {
  Write-Output "DRY RUN: would run 'kaggle kernels push -p $staging' then 'kaggle kernels status $($meta.id)'"
  return
}

# The CLI reads the token from KAGGLE_API_TOKEN (per-account files, never committed).
$env:KAGGLE_API_TOKEN = (Get-Content -LiteralPath $tokenFile -Raw).Trim()
try {
  kaggle kernels push -p $staging
  if ($LASTEXITCODE -ne 0) { throw "kaggle kernels push failed (exit $LASTEXITCODE)" }
  kaggle kernels status $meta.id
} finally {
  Remove-Item Env:\KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
}