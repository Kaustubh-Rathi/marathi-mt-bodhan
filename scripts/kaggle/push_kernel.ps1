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
  [Parameter(Mandatory = $true)][ValidateSet("prepare", "bodhan", "indictrans2", "eval", "ablation", "probe", "probe2", "probe3")][string]$Task,
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
    "ablation"    { "3" }
    "probe"       { "1" }
    "probe2"      { "2" }
    "probe3"      { "3" }
  }
}

# Explicit task -> account-username map; must match the owner prefix of the
# `id` field in kernel-metadata.<task>.json.
$TaskOwner = @{
  "bodhan"      = "kaustubhcrathi"
  "eval"        = "kaustubhcrathi"
  "ablation"    = "acajjhfh"
  "probe"       = "kaustubhcrathi"
  "indictrans2" = "dreamexcellence"
  "probe2"      = "dreamexcellence"
  "prepare"     = "acajjhfh"
  "probe3"      = "acajjhfh"
}
$AcctOwner = @{
  "1" = "kaustubhcrathi"
  "2" = "dreamexcellence"
  "3" = "acajjhfh"
}

$metadata = Join-Path $PSScriptRoot "kernel-metadata.$Task.json"
if (-not (Test-Path -LiteralPath $metadata)) { throw "missing metadata file: $metadata" }
$meta = Get-Content -LiteralPath $metadata -Raw | ConvertFrom-Json

# Abort if the metadata `id` owner does not match the chosen -Acct token's
# account (pushing another account's kernel id yields a confusing API error).
# Runs before -DryRun handling so dry runs validate too.
$metaOwner = ($meta.id -split "/")[0]
$acctUser = $AcctOwner[$Acct]
if (-not $acctUser) { throw "unknown -Acct '$Acct' (expected 1, 2, or 3)" }
if ($metaOwner -ne $acctUser) {
  throw "metadata id owner '$metaOwner' does not match -Acct $Acct account '$acctUser' (task '$Task' expects '$($TaskOwner[$Task])'); re-run with the correct -Acct or fix kernel-metadata.$Task.json"
}
if ($metaOwner -ne $TaskOwner[$Task]) {
  throw "metadata id owner '$metaOwner' does not match expected owner '$($TaskOwner[$Task])' for task '$Task'; fix kernel-metadata.$Task.json"
}

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

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) { throw "kaggle CLI not found on PATH (pip install kaggle); cannot push kernel" }

# The CLI reads the token from KAGGLE_API_TOKEN (per-account files, never committed).
$env:KAGGLE_API_TOKEN = (Get-Content -LiteralPath $tokenFile -Raw).Trim()
try {
  $pushOutput = kaggle kernels push -p $staging 2>&1 | Out-String
  $pushOutput.Trim() | Write-Output
  if ($LASTEXITCODE -ne 0) { throw "kaggle kernels push failed (exit $LASTEXITCODE)" }

  # Kaggle derives the kernel slug from the TITLE, which may differ from
  # metadata "id"; the push output prints the canonical URL, so resolve the real
  # slug from it (fall back to the metadata id).
  $slug = $meta.id
  $match = [regex]::Match($pushOutput, "kaggle\.com/code/([^\s/]+/[^\s/]+)")
  if ($match.Success) { $slug = $match.Groups[1].Value.TrimEnd('.') }
  if ($slug -ne $meta.id) {
    Write-Output "note        : kernel slug is '$slug' (title-derived), not metadata id '$($meta.id)'"
  }
  # The metadata owner was validated pre-push, but the CLI may be
  # authenticated as a DIFFERENT user (e.g. a stale legacy kaggle.json, which
  # outranks nothing and KAGGLE_API_TOKEN only the new CLI honours). A kernel
  # pushed under the wrong account cannot see this task's private datasets and
  # fails late and confusingly, so abort here instead of tailing it.
  $slugOwner = ($slug -split "/")[0]
  if ($slugOwner -ne $acctUser) {
    throw "pushed kernel owner '$slugOwner' does not match -Acct $Acct account '$acctUser' (slug '$slug'); the kaggle CLI authenticated as the wrong user - check $tokenFile / KAGGLE_API_TOKEN"
  }
  kaggle kernels status $slug
  Write-Output "logs        : kaggle kernels logs $slug"
  Write-Output "output      : kaggle kernels output $slug -p artifacts"
} finally {
  Remove-Item Env:\KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
}