set -e

[ "$1" == "--inner" ] && {
    # NVCUVID_LIBRARY=$PWD/Video_Codec_SDK_9.1.23/Lib/linux/stubs/x86_64/libnvcuvid.so
    # NVENCODE_LIBRARY=$PWD/Video_Codec_SDK_9.1.23/Lib/linux/stubs/x86_64/libnvidia-encode.so
    echo LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=$PWD/VideoProcessingFramework/install/bin:${LD_LIBRARY_PATH}
    # python3.6 SampleDecode.py 0 big_buck_bunny_1080p_h264.mov big_buck_bunny_1080p_h264.nv12
    cp $PWD/VideoProcessingFramework/install/bin/PyNvCodec.cpython-36m-x86_64-linux-gnu.so .
    python3.6 my.py
    exit 0
}

IMAGE=andrey-task1
docker build -t ${IMAGE} .

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run -it --runtime nvidia -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
    --ipc=host \
    --network host \
    --security-opt apparmor:unconfined \
    --volume=$XSOCK:$XSOCK:rw \
    --volume=$XAUTH:$XAUTH:rw \
    --env="XAUTHORITY=${XAUTH}" \
    --env="DISPLAY" \
    -v $PWD:/cdir \
    -w /cdir \
    ${IMAGE} bash $0 --inner

