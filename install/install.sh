#!/bin/bash
MERIDIAN_DIR="$(cd "$(dirname "$0")" && pwd)/.."
WEIGHTS_DIR="${MERIDIAN_WEIGHTS:-$MERIDIAN_DIR/weights}"
pushd $MERIDIAN_DIR

# pip install .
# pip install --no-build-isolation git+https://github.com/CASIA-IVA-Lab/FastSAM.git@4d153e9

# Install the cross-view place-recognition package (frame_descriptor:
# meridian-vpr) and copy its weights next to the others, where
# segmenter.vpr_ckpt looks for them.
git submodule update --init third_party/vpr
pip install -e third_party/vpr
mkdir -p $WEIGHTS_DIR
cp third_party/vpr/weights/*.pt $WEIGHTS_DIR/

# Install CLIPPER
mkdir third_party/clipper/build
cd third_party/clipper/build
cmake .. && make && make pip-install

# Download weights
cd $WEIGHTS_DIR
gdown 'https://drive.google.com/uc?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv'

popd
