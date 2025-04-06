#!/usr/bin/env zsh
# C code (EOCFI)
cd lime_tbx/business/eocfi_adapter/eocfi_c
cp MakefileDarwin.mak Makefile
make
cd ../../../..
# python code
# python3.9 is the manually installed, we avoid using the builtin one
rm -rf lime_tbx.egg-info dist build
python3.9 -m build
rm -rf .venv
python3.9 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install PySide2~=5.15
pyinstaller lime_tbx.spec
cd deployment/installer && ./build_mac_installer.sh
