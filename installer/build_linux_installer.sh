mkdir -p linux/installer_files
[ ! -d "linux/installer_files/kernels" ] && cp -r ../kernels linux/installer_files
[ ! -d "linux/installer_files/eocfi_data" ] && cp -r ../eocfi_data linux/installer_files
[ -d "linux/installer_files/LimeTBX" ] && rm -rf linux/installer_files/LimeTBX
[ -d "linux/lime_installer.zip" ] && rm -f linux/lime_installer.zip
cp -r ../dist/LimeTBX linux/installer_files
cp ../lime_tbx/gui/assets/lime_logo.png linux/installer_files
cp linux/limetbx.desktop linux/installer_files
zip -r linux/lime_installer.zip linux/installer_files linux/lime_installer.sh linux/lime_uninstaller.sh
