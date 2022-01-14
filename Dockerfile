FROM balenalib/amd64-ubuntu:focal
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /usr/src/app

RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash -\
  &&apt-get update && apt-get install -y --no-install-recommends \
  curl \
  wget \
  git \
  gnupg2 \
  apt-transport-https \
  gnupg \
  ca-certificates \
  lsb-release \
  python3-dev \
  cmake \
  python3-pip \
  nodejs \
  build-essential \
  python3-opencv

RUN wget https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021
RUN apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2021 && echo "deb https://apt.repos.intel.com/openvino/2021 all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2021.list
RUN apt-get update && apt-get install -y \
  intel-openvino-dev-ubuntu20-2021.3.394

RUN pip3 install numpy scipy pyyaml 
RUN echo "source /opt/intel/openvino_2021.3.394/bin/setupvars.sh" >> ~/.bashrc

RUN mkdir /opt/intel/openvino_2021.3.394/deployment_tools/open_model_zoo/demos/models && cd /opt/intel/openvino_2021/deployment_tools/tools/model_downloader

COPY ./models /opt/intel/openvino_2021.3.394/deployment_tools/open_model_zoo/demos/models/

SHELL ["/bin/bash", "-c"]

RUN source /opt/intel/openvino_2021.3.394/bin/setupvars.sh \
  && npm i inference-engine-node

ARG OpenCV_Version=4.2.0

# Download OpenCV
RUN cd / && curl https://codeload.github.com/opencv/opencv/tar.gz/${OpenCV_Version} --output ./opencv.tar.gz && tar xzf ./opencv.tar.gz \
    && mv /opencv-${OpenCV_Version} /opencv && \
    curl https://codeload.github.com/opencv/opencv_contrib/tar.gz/${OpenCV_Version} --output ./opencv_contrib.tar.gz && tar xzf ./opencv_contrib.tar.gz \
    && mv /opencv_contrib-${OpenCV_Version} /opencv_contrib && \
# Build OpenCV
    mkdir -p /opencv-build && cd /opencv-build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
        -D BUILD_TESTS=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D CMAKE_SHARED_LINKER_FLAGS=-latomic \
        -D BUILD_EXAMPLES=OFF \
        -DPYTHON3_EXECUTABLE=/usr/local/bin/python3.6 \
        –DPYTHON_INCLUDE_DIR=/usr/local/include/python3.6m \
        –DPYTHON_LIBRARY=/usr/local/lib/libpython3.6m.so \
        /opencv && \
    make --jobs=$(nproc --all) && \
    make install && \
    cd / ; rm -rf /opencv-build /opencv /opencv-contrib

ENV OpenCV_DIR=/usr/local/lib/cmake/opencv4

#RUN npm install opencv

CMD ["/bin/bash"]
