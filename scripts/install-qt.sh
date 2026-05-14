#!/usr/bin/env bash
# Install Qt 6 via aqtinstall and add QT_ROOT_DIR to your shell profile.
# Run once on a new machine; afterwards cmake --preset clang works without extra args.
#
# Usage: bash scripts/install-qt.sh [install_dir]

set -euo pipefail

QT_VERSION="6.9.2"
QT_ARCH="clang_64"
INSTALL_DIR="${1:-$HOME/Qt}"
QT_ROOT="$INSTALL_DIR/$QT_VERSION/macos"

# 1. Ensure aqt is available
if ! command -v aqt &>/dev/null; then
    echo "Installing aqtinstall via uv tool..."
    uv tool install aqtinstall
fi

# 2. Install Qt if not already present
if [ -f "$QT_ROOT/lib/cmake/Qt6/Qt6Config.cmake" ]; then
    echo "Qt $QT_VERSION already installed at $QT_ROOT"
else
    echo "Installing Qt $QT_VERSION ($QT_ARCH) to $INSTALL_DIR ..."
    aqt install-qt mac desktop "$QT_VERSION" "$QT_ARCH" -O "$INSTALL_DIR"
fi

# 3. Persist QT_ROOT_DIR in shell profile
EXPORT_LINE="export QT_ROOT_DIR=\"$QT_ROOT\""
PROFILE="${HOME}/.zshrc"
[ -n "${BASH_VERSION:-}" ] && PROFILE="${HOME}/.bashrc"

if grep -qF "QT_ROOT_DIR" "$PROFILE" 2>/dev/null; then
    echo "QT_ROOT_DIR already present in $PROFILE"
else
    echo "" >> "$PROFILE"
    echo "# Qt (set by scripts/install-qt.sh)" >> "$PROFILE"
    echo "$EXPORT_LINE" >> "$PROFILE"
    echo "Added QT_ROOT_DIR to $PROFILE — restart your terminal or run: source $PROFILE"
fi

# Export into the current session so cmake works immediately
export QT_ROOT_DIR="$QT_ROOT"
echo "QT_ROOT_DIR is available in this session: $QT_ROOT_DIR"
