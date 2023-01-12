#!/bin/bash
assets="installer_files"
dst="/opt/esa/LimeTBX"
user_home=$(bash -c "cd ~$(printf %q $SUDO_USER) && pwd")
local_appdata="$user_home/.LimeTBX"
executable="LimeTBX.exe"
x11_executable="LimeTBX.sh"
bin_path="/usr/bin"
command_name="lime"
desktop_applications="/usr/share/applications"
mkdir -p $local_appdata/kernels
mkdir -p $local_appdata/eocfi_data
mkdir -p $local_appdata/coeff_data
chmod 777 $local_appdata
chmod 777 $local_appdata/kernels
chmod 777 $local_appdata/eocfi_data
chmod 777 $local_appdata/coeff_data
mkdir -p $dst
cp -r $assets/* $dst
chmod 777 $dst/coeff_data/versions
chmod 777 $dst/coeff_data
printf "#!/usr/bin/env sh\nGTK_THEME=Adwaita XDG_SESSION_TYPE=x11 GDK_BACKEND=x11 $dst/LimeTBX/$executable \$@" >$dst/LimeTBX/$x11_executable
ln -s $dst/LimeTBX/$x11_executable $bin_path
ln -s $dst/limetbx.desktop $desktop_applications
mv $bin_path/$x11_executable $bin_path/$command_name
chmod -R +rx $dst
echo "LIME Toolbox installed successfully."