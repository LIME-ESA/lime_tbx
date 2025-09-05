#!/usr/bin/env zsh
set -euo pipefail

MODE="pkg"

usage() {
  cat <<EOF
Usage: $0 [--c | --pkg | --all]

  --c     Only compile EO-CFI C code (macOS/Darwin)
  --pkg   Only package Python app (without compiling EO-CFI C code) [default]
  --all   Compile EO-CFI C code and package Python app
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --c) MODE="c"; shift ;;
      --pkg) MODE="pkg"; shift ;;
      --all) MODE="all"; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
  done
}

compile_c() {
  echo "[build] Compiling C code (EO-CFI) for macOS..."
  cd repo/lime_tbx/business/eocfi_adapter/eocfi_c
  cp MakefileDarwin.mak Makefile
  make
  cd ../../../../..
}

package_python() {
  echo "[build] Packaging Python app for macOS..."
  cd repo
  rm -rf lime_tbx.egg-info dist build
  # python3.9 is the manually installed, we avoid using the builtin one
  python3.9 -m build
  rm -rf .venv
  python3.9 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  .venv/bin/pip install "PySide2~=5.15"
  export MACOSX_DEPLOYMENT_TARGET=10.15
  pyinstaller lime_tbx.spec
  cd deployment/installer
  ./build_mac_installer.sh
  cd ../../..
}

main() {
  parse_args "$@"

  case "$MODE" in
    c)
      compile_c
      ;;
    pkg)
      # Skipping C compilation is done to avoid dirty tag version.
      echo "WARNING: Skipping C compilation during packaging"
      echo "C code is NOT compiled in this package build!"
      package_python
      ;;
    all)
      compile_c
      package_python
      ;;
  esac
}

main "$@"
