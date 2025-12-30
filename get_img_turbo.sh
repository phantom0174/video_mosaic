mkdir -p libs_build
cd libs_build

wget https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.3.tar.gz -O libjpeg-turbo-3.1.3.tar.gz
tar -xzvf libjpeg-turbo-3.1.3.tar.gz
cd libjpeg-turbo-3.1.3

cmake -G"Unix Makefiles" -DCMAKE_INSTALL_PREFIX=../../mosaic_libs -DWITH_JPEG8=1 .
make -j$(nproc)
make install

cd ../..
echo "libjpeg-turbo installed to $(pwd)/mosaic_libs"
