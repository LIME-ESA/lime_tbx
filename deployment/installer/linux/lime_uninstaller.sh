#!/bin/bash
assets="installer_files"
dst="/opt/esa/LimeTBX"
user_home=$(bash -c "cd ~$(printf %q $SUDO_USER) && pwd")
local_appdata="$user_home/.LimeTBX"
executable="LimeTBX.exe"
bin_path="/usr/bin"
command_name="lime"
desktop_applications="/usr/share/applications"
rm -rf $local_appdata
rm $bin_path/$command_name
rm $desktop_applications/limetbx.desktop
rm -rf $dst
echo "LIME Toolbox uninstalled successfully."