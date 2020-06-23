set -e
VIDEO_FILE=big_buck_bunny_1080p_h264.mov
[ "$1" == "--inside" ] && {
    python3 -m pip install -r requirements.txt
    # NVCUVID_LIBRARY=$PWD/Video_Codec_SDK_9.1.23/Lib/linux/stubs/x86_64/libnvcuvid.so
    # NVENCODE_LIBRARY=$PWD/Video_Codec_SDK_9.1.23/Lib/linux/stubs/x86_64/libnvidia-encode.so
    echo LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=$PWD/VideoProcessingFramework/install/bin:${LD_LIBRARY_PATH}
    # python3.6 SampleDecode.py 0 big_buck_bunny_1080p_h264.mov big_buck_bunny_1080p_h264.nv12
    cp $PWD/VideoProcessingFramework/install/bin/PyNvCodec.cpython-36m-x86_64-linux-gnu.so .
    python3.6 my.py $VIDEO_FILE
    exit 0
}

IMAGE=andrey-task1
# docker build -t ${IMAGE} .

# XSOCK=/tmp/.X11-unix
# XAUTH=/tmp/.docker.xauth
# touch $XAUTH
# xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
#    --volume=$XSOCK:$XSOCK:rw \
#    --volume=$XAUTH:$XAUTH:rw \

mkdir -p .cache/pip || true

docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
    --ipc=host \
    --network host \
    --security-opt apparmor:unconfined \
    --env="XAUTHORITY=${XAUTH}" \
    --env="DISPLAY" \
    -v $PWD/.cache/pip:/home/ubuntu/.cache/pip \
    -v $(readlink -f $VIDEO_FILE):/cdir/$VIDEO_FILE \
    -v $(readlink -f VideoProcessingFramework):/cdir/VideoProcessingFramework \
    -v $PWD:/cdir \
    -w /cdir \
    ${IMAGE} bash $0 --inside

