#!/usr/bin/env bash
echo "Building lime .deb package."
# 1.0.2 Comment so the search for the version a.b.c appears (here it is a.b-c)
version="1.0-2"
# 1.0.2 Change it in the control file too!
name="lime_"$version
[ -d $name ] && rm -rf $name
[ -f $name".deb" ] && rm -f $name".deb"
mkdir $name
user_home=$(bash -c "cd ~$(printf %q $USER) && pwd")
local_appdata="$name$user_home/.LimeTBX"
mkdir -p $local_appdata/kernels
mkdir -p $local_appdata/eocfi_data
mkdir -p $local_appdata/coeff_data
chmod 777 $local_appdata
chmod 777 $local_appdata/kernels
chmod 777 $local_appdata/eocfi_data
chmod 777 $local_appdata/coeff_data
mkdir -p $name/opt/esa/LimeTBX
cp -r ../../kernels $name/opt/esa/LimeTBX
cp -r ../../eocfi_data $name/opt/esa/LimeTBX
cp -r ../../coeff_data $name/opt/esa/LimeTBX
cp ../../lime_tbx/gui/assets/lime_logo.png $name/opt/esa/LimeTBX
cp -r ../../dist/LimeTBX $name/opt/esa/LimeTBX
cp ../linux/limetbx.desktop $name/opt/esa/LimeTBX
mkdir -p $name/usr/share/applications
mkdir -p $name/usr/bin
ln -s /opt/esa/LimeTBX/limetbx.desktop $name/usr/share/applications
printf "#!/usr/bin/env sh\nGTK_THEME=Adwaita XDG_SESSION_TYPE=x11 GDK_BACKEND=x11 /opt/esa/LimeTBX/LimeTBX/LimeTBX.exe \$@" >$name/opt/esa/LimeTBX/LimeTBX/LimeTBX.sh
ln -s /opt/esa/LimeTBX/LimeTBX/LimeTBX.sh $name/usr/bin/lime
chmod 777 $name/opt/esa/LimeTBX/coeff_data/versions
chmod 777 $name/opt/esa/LimeTBX/coeff_data
chmod 777 $name/opt/esa/LimeTBX/coeff_data/interp_settings.yml
mkdir $name/DEBIAN
cp control $name/DEBIAN
cp postinst $name/DEBIAN
chmod -R +rx $name/opt/esa/LimeTBX
dpkg-deb --build $name