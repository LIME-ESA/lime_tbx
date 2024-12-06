cd repo
:: C code (EOCFI)
::cd lime_tbx\eocfi_adapter\eocfi_c
::copy make.mak Makefile
::.\..\..\..\deployment\windows_nmake.bat
::cd ..\..\..
:: python code
rd /s /q .venv
python -m venv .venv
.venv\scripts\python -m pip install --upgrade setuptools
.venv\scripts\python -m pip install --upgrade pip
.venv\scripts\pip install wheel
.venv\Scripts\pip install -r requirements.txt
rd /s /q dist build
python -m PyInstaller lime_tbx.spec
del "installer\windows\LimeTBX installer.exe"
cd installer\windows\
iscc inno_installer_builder.iss