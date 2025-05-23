#!/bin/bash
apt update
apt install libmpfr-dev libgmp3-dev libmpc-dev -y
apt install libevent-dev -y
apt install strace perftest -y
apt install gcc-12 g++-12 -y
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/x86_64-linux-gnu-g++-12 12
cd /usr/lib/x86_64-linux-gnu/
ln -sf libibverbs.so.1.14.39.0 libibverbs.so
ln -sf librdmacm.so.1.3.39.0 librdmacm.so
ln -sf libibumad.so.3.2.39.0 libibumad.so
pip install netifaces
