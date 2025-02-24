@echo off

setlocal

@set SYSTEM=%1
REM Remove previous log file
set LOG_FILE=logs\%SYSTEM%_log.txt

if EXIST %LOG_FILE% (
   del /f %LOG_FILE%
)

REM ----------------------------------
REM WRITE LOG FILE: example_OS_log.txt
REM ----------------------------------

for /f "delims=" %%a in ('date /T') do (
  set MYDATE=%%a
)

echo Example log:      > %LOG_FILE%
echo ---------------   >> %LOG_FILE%
echo                   >> %LOG_FILE%
echo SYSTEM = %SYSTEM% >> %LOG_FILE%
echo DATE = %MYDATE%   >> %LOG_FILE%

set LOG_FILE_LOOP=%LOG_FILE%
set cfiset=(file_handling data_handling lib orbit pointing visibility)

set PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\vc\Auxiliary\Build;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build;C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin;libraries\%SYSTEM%;%PATH%

if "%SYSTEM%" == "WINDOWS32" (
  @call vcvarsall.bat x86
) else (
  @call vcvarsall.bat x64
)

@set WIN_DLL=shared
nmake /nologo /f make.mak clean
nmake /nologo /f make.mak %WIN_DLL% > logs\stdout.txt 2> logs\stderr.txt

:eof

endlocal