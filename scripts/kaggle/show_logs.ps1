# Fetch a kernel's current logs and print a readable tail (ASCII-safe).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/kaggle/show_logs.ps1 -Task bodhan [-Acct 1]
param(
  [Parameter(Mandatory = $true)][ValidateSet("prepare","bodhan","indictrans2","eval","ablation","probe","probe2","probe3")][string]$Task,
  [ValidateSet("1","2","3")][string]$Acct = "",
  [int]$Tail = 60
)
$acctMap = @{ "1" = "$HOME\.kaggle\access_token"; "2" = "$HOME\.kaggle\access_token_acct2"; "3" = "$HOME\.kaggle\access_token_acct3" }
$taskAcct = @{ "prepare"="3"; "bodhan"="1"; "indictrans2"="2"; "eval"="1"; "ablation"="3"; "probe"="1"; "probe2"="2"; "probe3"="3" }
if (-not $Acct) { $Acct = $taskAcct[$Task] }

$meta = Get-Content -LiteralPath (Join-Path $PSScriptRoot "kernel-metadata.$Task.json") -Raw | ConvertFrom-Json
$slug = $meta.id
$env:KAGGLE_API_TOKEN = (Get-Content -LiteralPath $acctMap[$Acct] -Raw).Trim()

$raw = (kaggle kernels logs $slug 2>$null | Out-String)
Remove-Item Env:\KAGGLE_API_TOKEN -ErrorAction SilentlyContinue

# Extract "data":"..." fragments and join; drop non-ASCII for console safety.
$sb = New-Object System.Text.StringBuilder
foreach ($m in [regex]::Matches($raw, '"data":"(.*?)","time"', 'Singleline')) {
  $s = $m.Groups[1].Value -replace '\\n', "`n" -replace '\\"', '"' -replace '\\\\', '\'
  [void]$sb.Append($s)
}
$text = $sb.ToString()
$clean = ($text.ToCharArray() | Where-Object { [int]$_ -lt 128 }) -join ''
Write-Output (($clean -split "`n" | Select-Object -Last $Tail) -join "`n")
