#!/bin/bash

echo "Installing apt-get dependencies"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get -f install -y g++ git

echo "Installing Caffe dependencies"
# Caffe dependencies
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install -y libboost-all-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev

# Go installation
echo "Downloading and installing Go"
cd
wget https://storage.googleapis.com/golang/go1.5.1.linux-amd64.tar.gz
tar -xzf go1.5.1.linux-amd64.tar.gz
sudo cp -r go /usr/local

# GASNet installation
echo "Downloading and installing GASNet"
cd
wget http://gasnet.lbl.gov/GASNet-1.24.2.tar.gz
tar -xzf GASNet-1.24.2.tar.gz GASNet-1.24.2
cd GASNet-1.24.2
./configure --prefix /usr/local/GASNet --enable-udp --disable-mpi --enable-par --enable-segment-fast --disable-segment-fast --disable-aligned-segments --disable-pshm --with-segment-mmap-max=1GB
make
make run-tests
sudo make install

cd
mkdir repos

# Setup directory
echo "Linking /vagrant to ~/repos/filter_legion"
cd ~/repos
ln -s /vagrant ./filter_legion

# Legion installation
echo "Cloning and installing Legion"
cd ~/repos
git clone https://github.com/StanfordLegion/legion.git
git checkout master

echo "export LG_RT_DIR=~/repos/legion/runtime" >> ~/.bashrc
echo "export PATH=/usr/local/go/bin:$PATH" >> ~/.bashrc
