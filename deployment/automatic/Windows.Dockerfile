#FROM mcr.microsoft.com/dotnet/framework/runtime:4.8-windowsservercore-ltsc2016
FROM mcr.microsoft.com/windows/servercore:ltsc2019
#FROM mcr.microsoft.com/windows/servercore:ltsc2022
# Set the working directory
WORKDIR /temp

# Install Chocolatey for package management
RUN powershell -NoProfile -ExecutionPolicy Bypass -Command \
    "Set-ExecutionPolicy Bypass -Scope Process; \
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12; \
    $env:chocolateyVersion = '1.4.3'; \
    iwr https://chocolatey.org/install.ps1 -UseBasicParsing | iex"

# Install git (python dep)
RUN choco install git nuget.commandline -y --no-progress --ignore-checksums
RUN choco install python39 -y --no-progress --ignore-checksums
RUN choco install innosetup -y --no-progress --ignore-checksums

SHELL ["cmd", "/S", "/C"]

# Install updated certs
RUN mkdir "C:\Program Files\Common Files\SSL"
RUN powershell -Command \
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; \
    Invoke-WebRequest -Uri https://curl.se/ca/cacert.pem -OutFile \"C:\Program Files\Common Files\SSL\cacert.pem\""
ENV SSL_CERT_FILE="C:\Program Files\Common Files\SSL\cacert.pem"

RUN python -m pip install --upgrade pip
RUN python -m pip install wheel
RUN python -m pip install pyinstaller
RUN python -m pip install build

# Not installing it Visual Studio Build Tools, nmake is not found even after installing them

WORKDIR "C:\\"
ENTRYPOINT .\repo\deployment\automatic\windows_build_script.bat
#docker build .. -t lime_compiler -f Windows.Dockerfile
#for %F in ("%cd%\..") do set dirname=%~dpF
#docker run -v %dirname%:C:\repo lime_compiler