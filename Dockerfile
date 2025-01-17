# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.5    (apt)
# jupyter       latest (pip)
# pytorch       latest (pip)
# ==================================================================

#FROM ubuntu:18.04
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04


MAINTAINER Nico Lutz nico@bakkenbaeck.no

ENV LANG C.UTF-8

# ==================================================================
# Copy dir
# ------------------------------------------------------------------
WORKDIR /g2t
COPY . /g2t

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------
#
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        curl \
        unzip \
        unrar \
        && \
#
#    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
#    cd ~/cmake && \
#    ./bootstrap && \
#    make -j"$(nproc)" install && \
#
#
## ==================================================================
## python
## ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.5 \
        python3.5-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.5 ~/get-pip.py && \
    ln -s /usr/bin/python3.5 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.5 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-learn \
        matplotlib \
        Cython \
        && \

# ==================================================================
# jupyter & requirements
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyter \
        && \
    $PIP_INSTALL \
        -r \
        requirements.txt \
        && \
    $PIP_INSTALL \
        -r \
        requirements.opt.txt \
        && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN python3 -m spacy download de_core_news_md

#RUN git clone https://github.com/DuyguA/DEMorphy ~/DEMorphy
#RUN $PIP_INSTALL ~/DEMorphy
#RUN wget https://github.com/DuyguA/DEMorphy/blob/master/demorphy/data/words.dg
#RUN cp words.dg /env/lib/python3.6/site-packages/demorphy/data 


# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN git clone https://github.com/cmu-mtlab/meteor eval_tools/
RUN git clone https://github.com/jhclark/tercom eval_tools/

EXPOSE 8888
EXPOSE 6006
RUN ["/bin/bash"]
