#!/usr/bin/env sh
cd repo
# C code (EOCFI)
cd lime_tbx/eocfi_adapter/eocfi_c
cp MakefileLinux.mak Makefile
make
cd ../../..
# python code
rm -rf .venv
python3 -m venv .venv
.venv/bin/pip install wheel
.venv/bin/pip install -r requirements.txt
rm -rf dist build
pyinstaller lime_tbx.spec
rm -rf installer/linux/installer_files && rm installer/linux/lime_installer.zip && rm -rf installer/debian/lime_*
cd installer && ./build_linux_installer.sh
cd debian && ./build_deb.sh