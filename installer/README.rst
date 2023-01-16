==========
Installers
==========

Windows
=======

The installer is being constructed in a virtual machine executing Windows 10.

Linux
=====

The installer is being constructed in a virtual machine executing Ubuntu 16.

In order to install python 3.8 in Ubuntu 16, the following can be executedss::

    sudo apt update
    sudo apt upgrade
    sudo add-apt-repository ppa:0k53d-karl-f830m/openssl
    sudo apt install openssl
    sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
    sudo apt install libbz2-dev
    wget https://www.python.org/ftp/python/3.8.16/Python-3.8.16.tar.xz
    tar -xvf Python-3.8.16.tar.xz
    cd Python-3.8.16
    ./configure --enable-shared --enable-optimizations
    make
    sudo make altinstall
    cd
    sudo cp /usr/local/lib/libpython3.8.so.1.0 /usr/lib
    echo "alias python3=python3.8" > .bash_aliases
    source .bashrc
    # Installing modern fonts libraries
    sudo apt install libpng16-16
    cd ~/Downloads
    wget http://archive.ubuntu.com/ubuntu/pool/main/f/freetype/libfreetype6_2.10.1-2_amd64.deb
    sudo dpkg -i libfreetype6*
    wget http://archive.ubuntu.com/ubuntu/pool/main/f/fontconfig/fontconfig-config_2.13.1-2ubuntu3_all.deb
    # other option: wget http://archive.ubuntu.com/ubuntu/pool/main/f/fontconfig/fontconfig-config_2.13.1-4.2ubuntu5_all.deb
    sudo dpkg -i fontconfig-co*
    wget http://archive.ubuntu.com/ubuntu/pool/main/f/fontconfig/libfontconfig1_2.13.1-2ubuntu3_amd64.deb
    sudo dpkg -i libfontconfig1_*
    wget http://archive.ubuntu.com/ubuntu/pool/main/f/fontconfig/fontconfig_2.12.6-0ubuntu2_amd64.deb
    sudo dpkg -i fontconfig_2.12.6-0*
    # More libraries
    sudo apt install libxcb-xinerama0
    sudo apt install libglib2.0-dev # maybe it's not needed

Mac
===

The installer is currently being compiled in a virtual machine executing macOS 12 Monterey.
