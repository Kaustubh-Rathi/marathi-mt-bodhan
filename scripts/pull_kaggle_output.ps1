# Pull a Kaggle kernel's output to local staging.
# Usage: .\scripts\pull_kaggle_output.ps1 -Slug <owner/slug> -Acct <1|2|3> [-OutDir <dir>]
param(
  [Parameter(Mandatory=$true)][string]$Slug,
  [ValidateSet("1","2","3")][string]$Acct = "1",
  [string]$OutDir = "D:\Assignment\marathi-mt-bodhan\artifacts"
)
$tokenFile = switch ($Acct) {
  "1" { "$HOME\.kaggle\access_token" }
  "2" { "$HOME\.kaggle\access_token_acct2" }
  "3" { "$HOME\.kaggle\access_token_acct3" }
}
$env:KAGGLE_API_TOKEN = (Get-Content -LiteralPath $tokenFile -Raw).Trim()
kaggle kernels output $Slug -p $OutDir
Remove-Item Env:\KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
Write-Output "Pulled $Slug (acct $Acct) -> $OutDir"
