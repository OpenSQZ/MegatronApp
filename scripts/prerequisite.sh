#!/bin/bash
apt update
apt install libmpfr-dev libgmp3-dev libmpc-dev -y
apt install libevent-dev -y
apt install strace perftest -y
cd /usr/lib/x86_64-linux-gnu/
ln -sf libibverbs.so.1.14.50.0 libibverbs.so
ln -sf librdmacm.so.1.3.50.0 librdmacm.so
ln -sf libibumad.so.3.2.50.0 libibumad.so
pip install netifaces
