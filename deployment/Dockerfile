FROM ubuntu:16.04
WORKDIR /usr/src/app
RUN apt update
# Basic pkgs for scripts
RUN apt install wget zip -y
# Libs for python compilation
RUN apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev -y
RUN apt install libbz2-dev libgl1 libsm6 libxext6 libglib2.0-0 libxrender1 rsyslog-gssapi -y
# Libs for running pyinstaller & lime
RUN apt install libsqlite3-dev libdbus-1-3 libxkbcommon-x11-0 libfontconfig1 libwayland-cursor0 libwayland-client0 libwayland-egl1 -y
RUN apt install libxcb-shm0 libxcb-randr0 libxcb-image0 libxcb-render-util0 libxcb-keysyms1 libxcb-xinerama0 libxcb-icccm4 libxcb-xfixes0 libxcb-shape0 libxcb-render0 libxcb-util1 -y
RUN apt install libcairo-gobject2 libpango-1.0-0 libpangocairo-1.0-0 libcairo2 libgtk-3-0 libxcomposite-dev libgtk-3-dev libatk1.0-0 -y
# python 3.8
RUN wget https://www.python.org/ftp/python/3.8.20/Python-3.8.20.tgz
RUN tar xzf Python-3.8.20.tgz
RUN cd Python-3.8.20 && ./configure --enable-shared && make && make install
# --enable-optimizations
RUN cp /usr/local/lib/libpython3.8.so.1.0 /usr/lib
# SSL
RUN wget https://www.openssl.org/source/old/1.1.1/openssl-1.1.1q.tar.gz
RUN tar xzvf openssl-1.1.1q.tar.gz
RUN cd openssl-1.1.1q && ./config --prefix=/usr shared zlib-dynamic && make && make install
# pip & pyinstaller
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN python3 -m pip install pyinstaller
# updated fontconfig (supress warnings)
RUN apt-get install libpng16-16
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/f/freetype/libfreetype6_2.10.1-2_amd64.deb
RUN dpkg -i libfreetype6_2.10.1-2_amd64.deb
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/f/fontconfig/fontconfig-config_2.13.1-2ubuntu3_all.deb
RUN dpkg --force-all -i fontconfig-config_2.13.1-2ubuntu3_all.deb
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/f/fontconfig/libfontconfig1_2.13.1-2ubuntu3_amd64.deb
RUN dpkg -i libfontconfig1_2.13.1-2ubuntu3_amd64.deb

CMD ["./repo/deployment/ubuntu_build_script.sh"]
# docker build .. -t lime_compiler -f Dockerfile
# docker run -v $(dirname $(pwd)):/usr/src/app/repo lime_compiler