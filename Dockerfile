FROM pytorch/pytorch
RUN apt-get update
RUN apt-get install -y git
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
