cd repo
:: C code (EOCFI)
::cd lime_tbx\eocfi_adapter\eocfi_c
::copy make.mak Makefile
::.\..\..\..\deployment\windows_nmake.bat
::cd ..\..\..
:: python code
rd /s /q dist build lime_tbx.egg-info
python -m build
rd /s /q .venv
python -m venv .venv
.venv\scripts\pip install wheel
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install PySide6~=6.9
python -m PyInstaller lime_tbx.spec
del "deployment\installer\windows\LimeTBX installer.exe"
cd deployment\installer\windows\
iscc inno_installer_builder.iss