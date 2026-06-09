#!/bin/bash
MERIDIAN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."
pushd $MERIDIAN_DIR

pip install .
pip install --no-build-isolation git+https://github.com/CASIA-IVA-Lab/FastSAM.git@4d153e9

# Install CLIPPER
mkdir thirdparty/clipper/build
cd thirdparty/clipper/build
cmake .. && make && make pip-install

# Download weights
mkdir -p $MERIDIAN_DIR/weights
cd $MERIDIAN_DIR/weights
gdown 'https://drive.google.com/uc?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv'

popd
