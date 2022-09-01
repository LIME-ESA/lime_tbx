assets="installer_files"
dst="/opt/esa/LimeTBX"
user_home=$(bash -c "cd ~$(printf "%q" $SUDO_USER) && pwd")
local_appdata="$user_home/.LimeTBX"
echo $local_appdata
exit
executable="LIME_TBX"
bin_path="/usr/bin"
command_name="lime"
mkdir -p $local_appdata/kernels
mkdir -p $local_appdata/eocfi_data
sudo chmod 777 $local_appdata
sudo chmod 777 $local_appdata/kernels
sudo chmod 777 $local_appdata/eocfi_data
mkdir -p $dst
cp -r $assets/* $dst
ln -s $dst/$executable $bin_path
mv $bin_path/$executable $bin_path/$command_name
echo "LIME Toolbox installed successfully."