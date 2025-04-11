#!/usr/bin/env zsh
# setup script to setup the os/machine to be able to build
curl https://www.python.org/ftp/python/3.9.13/python-3.9.13-macos11.pkg -o /tmp/python-3.9.13-macos11.pkg
sudo installer -pkg /tmp/python-3.9.13-macos11.pkg -target /
echo 'export PATH="/Library/Frameworks/Python.framework/Versions/3.9/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
which python3
python3 -m pip install --upgrade pip
python3 -m pip install pyinstaller
