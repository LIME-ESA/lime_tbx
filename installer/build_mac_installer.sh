[ -d "mac/bundle" ] && rm -rf mac/bundle
[ -f "mac/lime.pkg" ] && rm -rf mac/lime.pkg
mkdir -p mac/bundle
cp -r ../dist/LimeTBX.app mac/bundle/LimeTBX.app
cp -r ../kernels mac/bundle/LimeTBX.app/Contents/Resources/kernels
cp -r ../eocfi_data mac/bundle/LimeTBX.app/Contents/Resources/eocfi_data
cp -r ../coeff_data mac/bundle/LimeTBX.app/Contents/Resources/coeff_data
version='1.0.2'
pkgbuild --install-location /Applications --root mac/bundle --identifier 'int.esa.LimeTBX' --scripts mac/scripts --version $version --component-plist mac/Components.plist mac/lime.pkg
