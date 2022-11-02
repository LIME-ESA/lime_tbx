mkdir -p mac/installer_files
[ ! -d "mac/installer_files/kernels" ] && cp -r ../kernels mac/installer_files
[ ! -d "mac/installer_files/eocfi_data" ] && cp -r ../eocfi_data mac/installer_files
cp ../dist/LIME_TBX mac/installer_files
cp ../lime_tbx/gui/assets/lime_logo.png mac/installer_files
zip -r mac/lime_installer.zip mac/installer_files mac/lime_installer.sh mac/lime_uninstaller.sh
