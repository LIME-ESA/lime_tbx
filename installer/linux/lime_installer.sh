#!/bin/bash
assets="installer_files"
dst="/opt/esa/LimeTBX"
user_home=$(bash -c "cd ~$(printf %q $SUDO_USER) && pwd")
local_appdata="$user_home/.LimeTBX"
executable="LIME_TBX"
bin_path="/usr/bin"
command_name="lime"
desktop_applications="/usr/share/applications"
mkdir -p $local_appdata/kernels
mkdir -p $local_appdata/eocfi_data
chmod 777 $local_appdata
chmod 777 $local_appdata/kernels
chmod 777 $local_appdata/eocfi_data
mkdir -p $dst
cp -r $assets/* $dst
ln -s $dst/$executable $bin_path
ln -s $dst/limetbx.desktop $desktop_applications
mv $bin_path/$executable $bin_path/$command_name
chmod -R +rx $dst
echo "LIME Toolbox installed successfully."