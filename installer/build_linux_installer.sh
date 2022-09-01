mkdir -p linux/installer_files
[ ! -d "linux/installer_files/kernels" ] && cp -r ../kernels linux/installer_files
[ ! -d "linux/installer_files/eocfi_data" ] && cp -r ../eocfi_data linux/installer_files
cp ../dist/LIME\ TBX linux/installer_files