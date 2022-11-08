#!/usr/bin/env sh
echo "Building lime .deb package."
echo "This script uses files from ../linux directory, so generate that installer first."
version="0.0-3"
name="lime_"$version
[ -d $name ] && rm -rf $name
[ -f $name".deb" ] && rm -f $name".deb"
mkdir $name
mkdir -p $name/opt/esa/LimeTBX
cp -r ../linux/installer_files/* $name/opt/esa/LimeTBX/
mkdir -p $name/usr/share/applications
mkdir -p $name/usr/bin
ln -s /opt/esa/LimeTBX/limetbx.desktop $name/usr/share/applications
ln -s /opt/esa/LimeTBX/LimeTBX/LimeTBX.exe $name/usr/bin/lime
mkdir $name/DEBIAN
cp control $name/DEBIAN
dpkg-deb --build $name