set -e
[ "$1" == "--inner" ] && {
    # CMAKE=$PWD/cmake/bin/cmake
    CMAKE=cmake
    VIDEO_CODEC_SDK_INCLUDE_DIR=$PWD/Video_Codec_SDK_9.1.23/include
    # NVCUVID_LIBRARY=$PWD/Video_Codec_SDK_9.1.23/Lib/linux/stubs/x86_64/libnvcuvid.so
    # NVENCODE_LIBRARY=$PWD/Video_Codec_SDK_9.1.23/Lib/linux/stubs/x86_64/libnvidia-encode.so
    NVCUVID_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvcuvid.so.1
    NVENCODE_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1

    rm -r VideoProcessingFramework/build || true
    rm -r VideoProcessingFramework/install || true
    mkdir VideoProcessingFramework/build || true
    cd VideoProcessingFramework/build
    $CMAKE .. -DVIDEO_CODEC_SDK_INCLUDE_DIR=${VIDEO_CODEC_SDK_INCLUDE_DIR} \
        -DNVCUVID_LIBRARY=${NVCUVID_LIBRARY} -DNVENCODE_LIBRARY=${NVENCODE_LIBRARY} \
        -DGENERATE_PYTHON_BINDINGS=1 \
        -DCMAKE_INSTALL_PREFIX=../install

    VERBOSE=1 make -j8
    make install
    exit 0
}

IMAGE=andrey-task1
docker build -t ${IMAGE} .

docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
    -v $PWD:/cdir \
    -w /cdir \
    ${IMAGE} bash $0 --inner

