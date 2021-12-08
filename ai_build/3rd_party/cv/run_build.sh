#/bin/bash
set -ex

BUILD_DIR="${PWD}/build/"
INSTALL_DIR="${PWD}/install/"

if [ -d $BUILD_DIR ]; then
  rm -rf $BUILD_DIR 
fi

if [ -d $INSTALL_DIR ]; then
  rm -rf $INSTALL_DIR 
fi

mkdir $BUILD_DIR

pushd $BUILD_DIR
  cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ..
  make VERBOSE=1
  make install
popd

echo "[DONE] CV BUILD"
