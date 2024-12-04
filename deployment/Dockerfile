FROM ubuntu:16.04

COPY . .
RUN apt update
RUN apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget -y
RUN apt install libbz2-dev libgl1 libsm6 libxext6 libglib2.0-0 libxrender1 rsyslog-gssapi -y
RUN wget https://www.python.org/ftp/python/3.8.20/Python-3.8.20.tgz
RUN tar xzf Python-3.8.20.tgz
RUN cd Python-3.8.20 && ./configure && make && make install
RUN python3 -m venv .venv
#RUN .venv/bin/activate/
RUN .venv/bin/pip install -r requirements.txt
RUN .venv/bin/python3 run.py