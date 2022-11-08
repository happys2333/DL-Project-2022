FROM anibali/pytorch:1.8.0-cuda11.6-ubuntu20.04
RUN sudo apt-get update
RUN sudo apt-get install git
RUN sudo ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN git clone https://github.com/happys2333/DL-Project-2022.git
