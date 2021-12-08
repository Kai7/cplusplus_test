#/bin/bash
set -ex

BUILD_DIR="${PWD}/build/"
INSTALL_DIR="${PWD}/install/"

THIRD_PARTY_DIR="${PWD}/3rd_party"

CV_INSTALL_PATH="${THIRD_PARTY_DIR}/cv/install"

if [ -d $BUILD_DIR ]; then
  rm -rf $BUILD_DIR 
fi

if [ -d $INSTALL_DIR ]; then
  rm -rf $INSTALL_DIR 
fi

mkdir $BUILD_DIR

pushd $BUILD_DIR
  cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
	-DCV_SDK_ROOT=$CV_INSTALL_PATH ..
  make VERBOSE=1
  make install
popd

echo "[DONE] AI BUILD"
