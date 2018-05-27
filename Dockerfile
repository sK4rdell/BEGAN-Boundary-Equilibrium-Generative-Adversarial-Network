FROM tensorflow/tensorflow:1.4.0-gpu-py3

RUN pip3 install numpy==1.13.3 scipy==0.19.1 imageio==2.2.0

# Set working directory to app
WORKDIR /app

# Copy everything under current dir into app folder
ADD . /app




