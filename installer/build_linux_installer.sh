mkdir -p linux/installer_files
[ ! -d "linux/installer_files/kernels" ] && cp -r ../kernels linux/installer_files
[ ! -d "linux/installer_files/eocfi_data" ] && cp -r ../eocfi_data linux/installer_files
cp ../dist/LIME_TBX linux/installer_files
cp ../lime_tbx/gui/assets/lime_logo.png linux/installer_files
cp linux/limetbx.desktop linux/installer_files