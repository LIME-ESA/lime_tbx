#!/usr/bin/env bash
set -euo pipefail

MODE="pkg"

usage() {
  cat <<EOF
Usage: $0 [--c | --pkg | --all]

  --c     Only compile EO-CFI C code
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
  echo "[build] Compiling C code (EO-CFI)..."
  cd repo/lime_tbx/business/eocfi_adapter/eocfi_c
  cp MakefileLinux.mak Makefile
  make
  cd ../../../../..
}

package_python() {
  echo "[build] Packaging Python app..."
  cd repo
  rm -rf lime_tbx.egg-info dist build
  python3.9 -m build
  rm -rf .venv
  python3.9 -m venv .venv
  .venv/bin/pip install wheel
  .venv/bin/pip install -r requirements.txt
  .venv/bin/pip install "PySide2~=5.15"
  pyinstaller lime_tbx.spec
  rm -rf deployment/installer/linux/installer_files \
         deployment/installer/linux/lime_installer.zip \
         deployment/installer/debian/lime_*
  cd deployment/installer
  ./build_linux_installer.sh
  cd debian
  ./build_deb.sh
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
