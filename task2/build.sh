set -e
[ "$1" == "--inside" ] && {
    # CMAKE=$PWD/cmake/bin/cmake
    # rm -r opencv/build || true
    rm -r opencv/install || true
    mkdir opencv/build || true
    cd opencv/build

    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=../install \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D CUDA_ARCH_BIN=7.5 \
    -D WITH_CUBLAS=1 \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D HAVE_opencv_python3=ON \
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \
    -D BUILD_EXAMPLES=ON ..

    VERBOSE=1 make -j8
    make install
    exit 0
}

IMAGE=andrey-task2
# docker build -t ${IMAGE} .

# docker run -it --runtime nvidia -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
    -v $PWD:/cdir \
    -w /cdir \
    ${IMAGE} bash $0 --inside

