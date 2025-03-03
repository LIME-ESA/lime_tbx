# Use Ubuntu 16.04
FROM ubuntu:16.04

WORKDIR /usr/src/app

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists
RUN apt update
#&& apt upgrade -y

# Install basic pkgs for scripts
RUN apt install wget zip -y
# Libs for python compilation
RUN apt install build-essential zlib1g-dev libncurses5-dev \
    libgdbm-dev libnss3-dev libssl-dev \
    libreadline-dev libffi-dev -y
RUN apt install libbz2-dev libgl1 libsm6 libxext6 \
    libglib2.0-0 libxrender1 rsyslog-gssapi -y
# Libs for pyinstaller & lime
RUN apt install libsqlite3-dev libdbus-1-3 libxkbcommon-x11-0 \
    libfontconfig1 libwayland-cursor0 libwayland-client0 libwayland-egl1  -y
RUN apt install libxcb-shm0 libxcb-randr0 libxcb-image0 libxcb-render-util0 \
    libxcb-keysyms1 libxcb-xinerama0 libxcb-icccm4 libxcb-xfixes0 libxcb-shape0 \
    libxcb-render0 libxcb-util1 -y
RUN apt install libcairo-gobject2 libpango-1.0-0 libpangocairo-1.0-0 libcairo2 \
    libgtk-3-0 libxcomposite-dev libgtk-3-dev libatk1.0-0 -y

#libxcb1 libfreetype6-dev curl ca-certificates

# Install OpenSSL 1.1.1w (without breaking system OpenSSL)
WORKDIR /usr/local/src
RUN wget https://www.openssl.org/source/old/1.1.1/openssl-1.1.1w.tar.gz
RUN tar xzvf openssl-1.1.1w.tar.gz
WORKDIR /usr/local/src/openssl-1.1.1w
RUN ./config --prefix=/usr/local/openssl --openssldir=/usr/local/openssl shared zlib-dynamic
RUN make -j$(nproc) && make install

# Update library paths
RUN echo "/usr/local/openssl/lib" >> /etc/ld.so.conf.d/openssl-1.1.1.conf && ldconfig

# Verify OpenSSL installation
RUN /usr/local/openssl/bin/openssl version -a

# Install Python 3.9 (linked to OpenSSL 1.1.1)
WORKDIR /usr/local/src
RUN wget https://www.python.org/ftp/python/3.9.21/Python-3.9.21.tgz
RUN tar xzf Python-3.9.21.tgz
WORKDIR /usr/local/src/Python-3.9.21
RUN ./configure --with-openssl=/usr/local/openssl --enable-shared --enable-optimizations
RUN make -j$(nproc) && make install

# Ensure Python can find shared libraries
RUN echo "/usr/local/lib" >> /etc/ld.so.conf.d/python3.9.conf && ldconfig

# Verify Python installation
RUN python3.9 --version
RUN python3.9 -c "import ssl; print(ssl.OPENSSL_VERSION)"

# Install pip and pyinstaller
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.9 get-pip.py
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install pyinstaller

# Suppress fontconfig warnings keeping an updated version
RUN apt-get install libpng16-16
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/f/freetype/libfreetype6_2.10.1-2_amd64.deb
RUN dpkg -i libfreetype6_2.10.1-2_amd64.deb
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/f/fontconfig/fontconfig-config_2.13.1-2ubuntu3_all.deb
RUN dpkg --force-all -i fontconfig-config_2.13.1-2ubuntu3_all.deb
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/f/fontconfig/libfontconfig1_2.13.1-2ubuntu3_amd64.deb
RUN dpkg -i libfontconfig1_2.13.1-2ubuntu3_amd64.deb
RUN apt-get -f install -y

WORKDIR /usr/src/app

CMD ["./repo/deployment/automatic/ubuntu_build_script.sh"]
# docker build ../.. -t lime_compiler -f Linux.Dockerfile
# docker run -v $(dirname $(dirname $(pwd))):/usr/src/app/repo lime_compiler