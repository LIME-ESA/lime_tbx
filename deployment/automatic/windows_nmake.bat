@echo off
:: This should work if nmake was installed correctly in the windows docker machine
set VSCMD="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
CALL %VSCMD%
where nmake