set -ex
[ "$1" == "--inside" ] && {

if false; then
    # build Eigen3
    mkdir eigen/build || true
    cd eigen/build
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=../install ..
    make -j8
    make install
fi

if false; then
# ARCH_BIN=7.2 # AGX Xavier
#ARCH_BIN=6.2 # Tx2
ARCH_BIN=6.1 # GTX 1050

    # build opencv
    cd /cdir
    # sudo cp Video_Codec_SDK_9.1.23/include/* /usr/local/cuda/include
    rm -rf opencv/install || true
    mkdir opencv/build || true
    cd opencv/build

    # NVCUVID_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvcuvid.so.1
    # NVENCODE_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1

    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_PREFIX_PATH="/cdir/eigen/install" \
    -D CMAKE_INSTALL_PREFIX=../install \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_CUDA=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1 \
    -D WITH_CUDNN=ON \
    -D CUDNN_VERSION='7.6' \
    -D CUDNN_INCLUDE_DIR='/usr/include/' \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D CUDA_ARCH_BIN=${ARCH_BIN} \
    -D WITH_CUBLAS=1 \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D HAVE_opencv_python3=ON \
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \
    -D BUILD_EXAMPLES=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_TBB=ON \
    ..

    VERBOSE=1 make -j8
    make install
    # -D WITH_NVCUVID=ON \
    # -D CUDA_nvcuvid_LIBRARY=${NVCUVID_LIBRARY} \
    # -D CUDA_nvcuvenc_LIBRARY=${NVENCODE_LIBRARY} \
    # -D BUILD_opencv_cudacodec=ON \

fi

if true; then
    cd /cdir
    #build tkDNN
    mkdir tkDNN/build || true
    cd tkDNN/build
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_PREFIX_PATH="/cdir/eigen/install;/cdir/opencv/install" \
    -D CUDA_nvinfer_LIBRARY="/usr/lib/x86_64-linux-gnu/libnvinfer.so" \
    -D CMAKE_INSTALL_PREFIX=../install ..
    make -j8

    # -D Eigen3_DIR=$PWD/eigen/install/share/eigen3/cmake \
fi

    exit 0
}

IMAGE=andrey-task3
# docker build -t ${IMAGE} .

docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
    -v $PWD:/cdir \
    -v $(readlink -f opencv):/cdir/opencv \
    -w /cdir \
    ${IMAGE} bash $0 --inside

    # -v $(readlink -f Video_Codec_SDK_9.1.23):/cdir/Video_Codec_SDK_9.1.23 \
