$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repoRoot

function Ensure-FileFromExample {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$ExamplePath
  )

  if (Test-Path -LiteralPath $Path) {
    return
  }
  if (-not (Test-Path -LiteralPath $ExamplePath)) {
    throw "Missing example file: $ExamplePath"
  }

  Copy-Item -LiteralPath $ExamplePath -Destination $Path
  Write-Host "Created $Path from $ExamplePath"
}

Ensure-FileFromExample -Path ".env" -ExamplePath ".env.example"
Ensure-FileFromExample -Path "frontend\\.env" -ExamplePath "frontend\\.env.example"

if (-not (Test-Path -LiteralPath ".venv")) {
  python -m venv .venv
}

$python = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r backend\\requirements.txt

Push-Location frontend
if (Test-Path -LiteralPath "node_modules") {
  npm install
} else {
  npm ci
}
Pop-Location

$backendCommand = "Set-Location -LiteralPath '$repoRoot'; .\\.venv\\Scripts\\python -m uvicorn app.main:app --app-dir backend --reload"
Write-Host "Starting backend in a new window..."
Start-Process -FilePath "powershell" -ArgumentList @("-NoExit", "-Command", $backendCommand)

Write-Host "Starting frontend in this window..."
Set-Location -LiteralPath (Join-Path $repoRoot "frontend")
npm run dev
