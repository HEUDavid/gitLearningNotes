#!/bin/bash
## prepare source
mkdir tools && mkdir wget && cd wget && wget http://mirrors.nju.edu.cn/gnu/texinfo/texinfo-6.7.tar.gz && wget http://mirrors.nju.edu.cn/gnu/gdb/gdb-7.9.tar.gz
tar -zxvf texinfo-6.7.tar.gz &&
tar -zxvf gdb-7.9.tar.gz &&
cd ../tools && mkdir gdb && mkdir texinfo &&
echo success

## 路径
## install texinfo
cd ~/wget/texinfo-6.7 && mkdir build-texinfo && cd build-texinfo &&
../configure --prefix=/data/home_ext/davidxiang/tools/texinfo && make -j8 && make install

## set path texinfo
# vim ~/.bashrc
export PATH=/data/home_ext/davidxiang/workspace/futu_practice/venv/bin:/data/home_ext/davidxiang/tools/texinfo/bin:/data/home_ext/davidxiang/tools/texinfo/lib:$PATH
export LD_LIBRARY_PATH=/data/home_ext/davidxiang/tools/texinfo/lib:$LD_LIBRARY_PATH


## install gdb
cd ~/wget/gdb-7.9/ && mkdir build-gdb && cd build-gdb &&
../configure --prefix=/data/home_ext/davidxiang/tools/gdb && make -j4 && make install

## set path
export PATH=/data/home_ext/davidxiang/workspace/futu_practice/venv/bin:/data/home_ext/davidxiang/tools/gdb/bin:/data/home_ext/davidxiang/tools/gdb/lib:/data/home_ext/davidxiang/tools/texinfo/bin:/data/home_ext/davidxiang/tools/texinfo/lib:$PATH
export LD_LIBRARY_PATH=/data/home_ext/davidxiang/tools/texinfo/lib:/data/home_ext/davidxiang/tools/gdb/lib:$LD_LIBRARY_PATH

# source .bashrc
