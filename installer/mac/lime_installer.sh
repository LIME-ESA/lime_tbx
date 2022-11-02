#!/bin/zsh
assets="installer_files"
dst="/Applications/LimeTBX"
user_home=$(bash -c "cd ~$(printf %q $SUDO_USER) && pwd")
local_appdata="$user_home/Library/Application Support/LimeTBX"
executable="LIME_TBX"
bin_path="/usr/local/bin"
command_name="lime"
mkdir -p $local_appdata/kernels
mkdir -p $local_appdata/eocfi_data
chmod 777 $local_appdata
chmod 777 $local_appdata/kernels
chmod 777 $local_appdata/eocfi_data
cp -r $assets/* $local_appdata
mv $local_appdata/$executable $dst
ln -s $dst $bin_path/$command_name
chmod +rx $dst
echo "LIME Toolbox installed successfully."
