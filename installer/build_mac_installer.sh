mkdir -p mac/installer_files
[ ! -d "mac/scripts/installer_files/kernels" ] && cp -r ../kernels mac/installer_files
[ ! -d "mac/scripts/installer_files/eocfi_data" ] && cp -r ../eocfi_data mac/installer_files
cp -r ../dist/LimeTBX.app mac/bundle
mv mac/installer_files/* mac/bundle/LimeTBX.app/Contents/Resources
#zip -r mac/lime_installer.zip mac/installer_files mac/lime_installer.sh mac/lime_uninstaller.sh
version='0.0.2'
pkgbuild --install-location /Applications --root mac/bundle --identifier 'int.esa.lime' --version $version mac/lime.pkg
