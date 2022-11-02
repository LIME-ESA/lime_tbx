#!/bin/zsh
assets="installer_files"
dst="/Applications/LimeTBX"
user_home=$(bash -c "cd ~$(printf %q $SUDO_USER) && pwd")
local_appdata="$user_home/Library/Application Support/LimeTBX"
executable="LIME_TBX"
bin_path="/usr/local/bin"
command_name="lime"
rm -rf $local_appdata
rm $bin_path/$command_name
rm $dst
echo "LIME Toolbox uninstalled successfully."
