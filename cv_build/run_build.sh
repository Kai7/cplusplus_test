#/bin/bash
set -ex

BUILD_DIR="${PWD}/build/"
INSTALL_DIR="${PWD}/install/"

DATA_HOST_PATH="/home/kai7/data"
TOOLCHAIN_PATH="${DATA_HOST_PATH}/wisecore_repo/cvitek_dev/host-tools"
HOST_TOOL_PATH="${TOOLCHAIN_PATH}/gcc/arm-cvitek-linux-uclibcgnueabihf"
#HOST_TOOL_PATH="${TOOLCHAIN_PATH}/gcc/arm-cvitek-linux-uclibcgnueabihf/bin"
#TT="/home/kai7/data/wisecore_repo/cvitek_dev/host-tools/gcc/arm-cvitek-linux-uclibcgnueabihf/bin/"

TOOLCHAIN_FILE="toolchain/toolchain-uclibc-linux.cmake"

THIRD_PARTY_DIR="${PWD}/3rd_party"
OPENCV_INSTALL_PATH="${THIRD_PARTY_DIR}/opencv"
if [ ! -d $OPENCV_INSTALL_PATH ]; then
  echo "OPENCV SDK Not Found."
  exit 1
fi

if [ -d $BUILD_DIR ]; then
  rm -rf $BUILD_DIR 
fi

if [ -d $INSTALL_DIR ]; then
  rm -rf $INSTALL_DIR 
fi

mkdir $BUILD_DIR

pushd $BUILD_DIR
  cmake -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE \
    -DTOOLCHAIN_ROOT_DIR=$HOST_TOOL_PATH \
	-DOPENCV_ROOT=$OPENCV_INSTALL_PATH \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ..
  make VERBOSE=1
  make install
popd

echo "[DONE] AI BUILD"
