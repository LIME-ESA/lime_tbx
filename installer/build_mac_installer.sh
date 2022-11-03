mkdir -p mac/temp_files
[ ! -d "mac/temp_files/kernels" ] && cp -r ../kernels mac/temp_files
[ ! -d "mac/temp_files/eocfi_data" ] && cp -r ../eocfi_data mac/temp_files
mkdir -p mac/bundle
cp -r ../dist/LimeTBX.app mac/bundle/LimeTBX.app
mv mac/temp_files/* mac/bundle/LimeTBX.app/Contents/Resources
rmdir mac/temp_files
#zip -r mac/lime_installer.zip mac/installer_files mac/lime_installer.sh mac/lime_uninstaller.sh
version='0.0.2'
#pkgbuild --install-location /Applications --root mac/bundle --identifier 'int.esa.LimeTBX' --version $version mac/lime.pkg
