# Install Qt 6 via aqtinstall and set QT_ROOT_DIR as a persistent user env var.
# Run once on a new machine; afterwards cmake --preset win64 works without extra args.
#
# Usage: pwsh scripts/install-qt.ps1 [-InstallDir <path>]

param(
    [string]$InstallDir = "C:\Qt"
)

$QT_VERSION  = "6.9.2"
$QT_ARCH     = "win64_msvc2022_64"
$QT_ROOT     = "$InstallDir\$QT_VERSION\msvc2022_64"

# 1. Ensure aqt is available
if (-not (Get-Command aqt -ErrorAction SilentlyContinue)) {
    Write-Host "Installing aqtinstall via uv tool..."
    uv tool install aqtinstall
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to install aqtinstall"; exit 1 }
}

# 2. Install Qt if not already present
if (Test-Path "$QT_ROOT\lib\cmake\Qt6\Qt6Config.cmake") {
    Write-Host "Qt $QT_VERSION already installed at $QT_ROOT"
} else {
    Write-Host "Installing Qt $QT_VERSION ($QT_ARCH) to $InstallDir ..."
    aqt install-qt windows desktop $QT_VERSION $QT_ARCH -O $InstallDir
    if ($LASTEXITCODE -ne 0) { Write-Error "Qt installation failed"; exit 1 }
}

# 3. Persist QT_ROOT_DIR for the current user so cmake --preset works without extra args
$current = [System.Environment]::GetEnvironmentVariable("QT_ROOT_DIR", "User")
if ($current -ne $QT_ROOT) {
    [System.Environment]::SetEnvironmentVariable("QT_ROOT_DIR", $QT_ROOT, "User")
    Write-Host "Set user env var QT_ROOT_DIR=$QT_ROOT"
    Write-Host "Restart your terminal (or re-source your profile) for the change to take effect."
} else {
    Write-Host "QT_ROOT_DIR already set to $QT_ROOT"
}

# Also export into the current session so cmake works immediately without a restart
$env:QT_ROOT_DIR = $QT_ROOT
Write-Host "QT_ROOT_DIR is available in this session: $env:QT_ROOT_DIR"
