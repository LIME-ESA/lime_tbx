#!/usr/bin/env sh
cd repo
# C code (EOCFI)
cd lime_tbx/business/eocfi_adapter/eocfi_c
cp MakefileLinux.mak Makefile
make
cd ../../../..
# python code
rm -rf .venv
python3.9 -m venv .venv
.venv/bin/pip install wheel
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install PySide2~=5.15
rm -rf dist build
pyinstaller lime_tbx.spec
rm -rf deployment/installer/linux/installer_files && rm deployment/installer/linux/lime_installer.zip && rm -rf deployment/installer/debian/lime_*
cd deployment/installer && ./build_linux_installer.sh
cd debian && ./build_deb.sh