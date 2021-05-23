## MP-SPDZ Installation Instructions

Tested against Ubuntu 18.04 and 20.04 based on git repo:
https://github.com/data61/MP-SPDZ/#requirements

Requirements:

```
sudo apt install automake build-essential git libboost-dev libboost-thread-dev libsodium-dev libssl-dev libtool m4 python texinfo yasm
```

#### Get MP-SPDZ

**Note:** Renaming the directory ensures it's compatible with the Selest scripts.

```
git clone --recurse-submodules https://github.com/data61/MP-SPDZ.git
mv MP-SPDZ MP_SPDZ_online
cd MP_SPDZ_online
```

#### MPIR

Note: Compile with C++ support using flag --enable-cxx when running configure

```
cd mpir
automake --add-missing
autoreconf -i
./configure --enable-cxx
make -j
make check
sudo make install
make installcheck
make clean
sudo ldconfig
cd ..
```

#### NTL (Number Theory Library)

**Note:** Use `libntl43` for Ubuntu 20.04.

```
sudo apt install libntl-dev libntl35
```

-or-

Get it from the source (not tested):

https://www.shoup.net/ntl/


**Note:** In case that the source code is cloned from the original repo, one can fetch the previous requirements along
with SimpleOT using `--recurse-submodules` and compile them in the subdirectories using the previous instructions.

## Installation

Run `make` to compile everything or simply compile required software separately. For example:

```
make -j mascot-party.x
```

Then compile the high level programs using the python script:

```
./compile.py -F 32 tutorial     # Compiles the tutorial for mod(prime) arithmetic using 64-bit integers
```

Test the tutorial
```
echo 1 2 3 4 > Player-Data/Input-P0-0
echo 1 2 3 4 > Player-Data/Input-P1-0
Scripts/mascot.sh tutorial
```

There are three computational domains that need to be specified during compile time. For more details check the [official
repo documentation](https://github.com/data61/MP-SPDZ).
