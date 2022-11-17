#!/usr/bin/env bash
echo "Building lime .deb package."
echo "This script uses files from ../linux directory, so generate that installer first."
# 0.0.4 Comment so the search for the version a.b.c appears (here it is a.b-c)
# Change it in the control file too!
version="0.0-4"
name="lime_"$version
[ -d $name ] && rm -rf $name
[ -f $name".deb" ] && rm -f $name".deb"
mkdir $name
user_home=$(bash -c "cd ~$(printf %q $SUDO_USER) && pwd")
local_appdata="$name$user_home/.LimeTBX"
mkdir -p $local_appdata/kernels
mkdir -p $local_appdata/eocfi_data
mkdir -p $local_appdata/coeff_data
chmod 777 $local_appdata
chmod 777 $local_appdata/kernels
chmod 777 $local_appdata/eocfi_data
chmod 777 $local_appdata/coeff_data
mkdir -p $name/opt/esa/LimeTBX
cp -r ../linux/installer_files/* $name/opt/esa/LimeTBX/
mkdir -p $name/usr/share/applications
mkdir -p $name/usr/bin
ln -s /opt/esa/LimeTBX/limetbx.desktop $name/usr/share/applications
ln -s /opt/esa/LimeTBX/LimeTBX/LimeTBX.exe $name/usr/bin/lime
chmod 777 $name/opt/esa/LimeTBX/coeff_data/versions
mkdir $name/DEBIAN
cp control $name/DEBIAN
dpkg-deb --build $name