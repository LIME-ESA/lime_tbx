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
.venv\Scripts\pip install wheel
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install PySide6~=6.9
python -m PyInstaller lime_tbx.spec
mkdir dist\LimeTBX\bin
> dist\LimeTBX\bin\lime.cmd echo @echo off
>> dist\LimeTBX\bin\lime.cmd echo "%%~dp0..\LimeTBX.exe" %%*
del "deployment\installer\windows\LimeTBX installer.exe"
cd deployment\installer\windows\
iscc inno_installer_builder.iss