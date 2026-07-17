# Push to GitHub using a Personal Access Token (no device OTP, no SSH).
#
# 1) Create a classic token:
#    https://github.com/settings/tokens/new
#    Scopes: check "repo"
# 2) Paste the token into this file ONLY (one line, no quotes):
#    C:\Users\Ibteshamul Haque\credit-risk-assessment\.github_pat
#    (this file is gitignored)
# 3) Run:
#    powershell -ExecutionPolicy Bypass -File scripts\push_with_pat.ps1
#
# Optional: set username if not titoatwork
#    $env:GITHUB_USER = "yourusername"

$ErrorActionPreference = "Stop"
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
Set-Location (Split-Path $PSScriptRoot -Parent)

$patFile = Join-Path (Get-Location) ".github_pat"
if (-not (Test-Path $patFile)) {
  Write-Host "Create file: $patFile" -ForegroundColor Yellow
  Write-Host "Put your GitHub Personal Access Token on the first line, save, re-run this script." -ForegroundColor Yellow
  Write-Host "Create token: https://github.com/settings/tokens/new  (scope: repo)" -ForegroundColor Cyan
  notepad $patFile
  exit 1
}

$token = (Get-Content $patFile -Raw).Trim()
if ($token.Length -lt 20) {
  Write-Host "Token file looks empty. Paste a valid PAT into .github_pat" -ForegroundColor Red
  exit 1
}

$user = $env:GITHUB_USER
if (-not $user) { $user = "titoatwork" }
$repo = "credit-risk-assessment"

$headers = @{
  Authorization = "Bearer $token"
  Accept = "application/vnd.github+json"
  "User-Agent" = "credit-risk-push"
  "X-GitHub-Api-Version" = "2022-11-28"
}

# Verify token
try {
  $me = Invoke-RestMethod -Uri "https://api.github.com/user" -Headers $headers
  $user = $me.login
  Write-Host "Authenticated as: $user" -ForegroundColor Green
} catch {
  Write-Host "Token invalid/unauthorized. Create a new classic PAT with 'repo' scope." -ForegroundColor Red
  throw
}

# Create repo if missing
try {
  Invoke-RestMethod -Uri "https://api.github.com/repos/$user/$repo" -Headers $headers | Out-Null
  Write-Host "Repo already exists."
} catch {
  Write-Host "Creating public repo $user/$repo ..."
  $body = @{
    name = $repo
    description = "Credit Risk Assessment: LR + XGBoost + SHAP (FastAPI + Streamlit) on Home Credit data"
    private = $false
    auto_init = $false
  } | ConvertTo-Json
  Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post -Headers $headers -Body $body -ContentType "application/json" | Out-Null
}

# Push via HTTPS with token (do not print URL with token)
git remote remove origin 2>$null
$remote = "https://$user`:$token@github.com/$user/$repo.git"
git remote add origin $remote
git push -u origin main
# Scrub token from remote URL after push
git remote set-url origin "https://github.com/$user/$repo.git"

Write-Host ""
Write-Host "SUCCESS: https://github.com/$user/$repo" -ForegroundColor Green

# Remove local PAT file for safety (optional)
# Remove-Item $patFile -Force
