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
# Removing the packaged libstdc++.so.6 library if the system's version is greater or equal
packaged_libstdcp=$dst/LimeTBX/libstdc++.so.6
sys_libstd="$(ldconfig -p | grep libstdc++.so.6 | tr ' ' '\n' | grep /)"
if [ -f $sys_libstd ]; then
    glibcversion=$(strings $packaged_libstdcp | grep GLIBCXX | tail -2 | head -1)
    strings $sys_libstd | grep -q $glibcversion && rm $packaged_libstdcp
fi
# /usr/lib/x86_64-linux-gnu is a debian-ism. In order to simulate it in other distros a link is created where those
# libraries actually are in those distros (/usr/lib64). This supresses some warnings, but if removed the functionalities
# wouldn't be affected.
[ ! -d "/usr/lib/x86_64-linux-gnu" ] && ln -s /usr/lib64 /usr/lib/x86_64-linux-gnu
echo "LIME Toolbox installed successfully."