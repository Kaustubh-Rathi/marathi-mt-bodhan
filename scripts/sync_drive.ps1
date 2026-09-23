# Push local staging to Google Drive via rclone.
# Usage: .\scripts\sync_drive.ps1 [-Source <dir>] [-Dest <rclone:path>] [-DryRun]
param(
  [string]$Source = "D:\Assignment\marathi-mt-bodhan\artifacts",
  [string]$Dest   = "gdrive:mr-mt-edu-2026/",
  [switch]$DryRun
)
$rclonePathFile = Join-Path $PSScriptRoot "rclone_path.txt"
if (Test-Path -LiteralPath $rclonePathFile) {
  $rclone = (Get-Content -LiteralPath $rclonePathFile -Raw).Trim()
} else {
  $rclone = "rclone"
}
$args = @("copy", $Source, $Dest, "--progress", "--create-empty-src-dirs", "--exclude", ".git/**")
if ($DryRun) { $args += "--dry-run" }
& $rclone @args
Write-Output "Done. Verify with: `"$rclone`" lsd $Dest"
