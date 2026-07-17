# Push credit-risk-assessment to GitHub via SSH (no device OTP).
# Prerequisites:
#   1. Add your public key at https://github.com/settings/keys
#      Public key file: %USERPROFILE%\.ssh\id_ed25519_github.pub
#   2. Create empty repo on GitHub named credit-risk-assessment (or set $RepoName)
#   3. Run this script from project (or: powershell -ExecutionPolicy Bypass -File scripts\push_to_github.ps1)

$ErrorActionPreference = "Stop"
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
Set-Location (Split-Path $PSScriptRoot -Parent)

$GitHubUser = $env:GITHUB_USER
if (-not $GitHubUser) {
  # Default from local git credentials history; change if wrong
  $GitHubUser = "titoatwork"
}
$RepoName = "credit-risk-assessment"
$sshUrl = "git@github.com:$GitHubUser/$RepoName.git"

Write-Host "Testing SSH to GitHub..." -ForegroundColor Cyan
ssh -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | Write-Host

git remote remove origin 2>$null
git remote add origin $sshUrl

Write-Host "Pushing main to $sshUrl ..." -ForegroundColor Cyan
git push -u origin main

Write-Host "Done: https://github.com/$GitHubUser/$RepoName" -ForegroundColor Green
