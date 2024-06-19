# Use an official Ubuntu as a parent image
FROM ubuntu:20.04

# Set the working directory
WORKDIR /root

# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libjpeg-dev \
    libtiff-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    wget \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    libgl1-mesa-glx \
    && apt-get clean

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Create and activate conda environment, install numpy and scipy
RUN conda create -n opencv_env python=3.9 numpy scipy -y && \
    conda clean -a -y

# Clone OpenCV repositories
RUN git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git

# Create build directory and configure the build
RUN cd opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D ENABLE_CXX11=ON \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_OPENGL=ON \
    -D WITH_OPENCL=ON \
    -D WITH_IPP=ON \
    -D WITH_CUDA=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_GTK=ON \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE=/opt/conda/envs/opencv_env/bin/python \
    -D BUILD_EXAMPLES=ON \
    -D BUILD_opencv_gapi=OFF \
    ..

# Compile and install OpenCV
RUN cd /root/opencv/build && make -j$(nproc) && make install && ldconfig

# Activate the environment by default
RUN echo "source activate opencv_env" > ~/.bashrc
ENV PATH /opt/conda/envs/opencv_env/bin:$PATH

# Set default command to bash
CMD ["bash"]

