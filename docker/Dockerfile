# Invoke with the following to enable Swift REPL
# build: docker build --rm -t netlib-dev:cuda .
# run:   docker run -it netlib-dev:cuda bash
# lldb/repl run: docker run --security-opt seccomp=unconfined -it netlib-dev:cuda bash
#
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# setup swift toolchain --------------------------------------------------------
ARG swift_dir=/usr/local/swift
ARG swift_package=swift-DEVELOPMENT-SNAPSHOT-2019-01-24-a-ubuntu18.04.tar.gz
ARG swift_url=https://swift.org/builds/development/ubuntu1804/swift-DEVELOPMENT-SNAPSHOT-2019-01-24-a/$swift_package

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
	clang \
	curl \
	gnupg2 \
	libicu-dev \
	libpython-dev \
	libncurses5-dev \
	libxml2 \
	wget

RUN mkdir $swift_dir \
	&& curl -SL $swift_url -o $swift_dir/$swift_package \
	&& curl -SsL $swift_url.sig -o $swift_dir/$swift_package.sig

RUN wget -q -O - https://swift.org/keys/all-keys.asc | gpg --import -

RUN gpg --verify $swift_dir/$swift_package.sig \
	&& rm $swift_dir/$swift_package.sig \
	&& tar xzf $swift_dir/$swift_package --strip-components=1 -C $swift_dir \
	&& rm $swift_dir/$swift_package

# netlib -----------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
	cmake \
	git \
	libbsd-dev \
	libzip-dev \
	libpng-dev \
	libjpeg-dev \
	liblmdb-dev \
	libblocksruntime0

# clone netlib source via HTTPS
ARG netlib_dir=/root/netlib
RUN git clone https://github.com/ewconnell/Netlib.git $netlib_dir

# bashrc -----------------------------------------------------------------------
RUN echo "export PATH=$swift_dir/usr/bin:${PATH}\ncd $netlib_dir" >> ~/.bashrc


