FROM ubuntu:20.04

# Avoid prompts from apt.
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3-pip python3-dev git build-essential cmake pkg-config \
    libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    gstreamer1.0-doc gstreamer1.0-tools libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libssl-dev libjpeg-dev libpng-dev libtiff-dev

# Install numpy which is a dependency for OpenCV Python bindings
RUN pip3 install numpy

# Clone OpenCV repos
RUN git clone https://github.com/opencv/opencv.git /opencv && \
    git clone https://github.com/opencv/opencv_contrib.git /opencv_contrib && \
    cd /opencv && git checkout 4.5.4 && \
    cd /opencv_contrib && git checkout 4.5.4

# Prepare build
RUN mkdir /opencv/build && cd /opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
    -D PYTHON_EXECUTABLE=$(which python3) \
    -D BUILD_EXAMPLES=ON \
    -D WITH_GSTREAMER=ON \
    -D BUILD_opencv_gapi=OFF \
    -D BUILD_opencv_python3=ON ..

# Build and install
RUN cd /opencv/build && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Check if cv2 is installed in the site-packages
RUN python3 -c "import site; print(site.getsitepackages())"

CMD ["python3"]
