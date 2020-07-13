# https://www.pyimagesearch.com/2020/02/10/opencv-dnn-with-nvidia-gpus-1549-faster-yolo-ssd-and-mask-r-cnn/
set -e
OPENCV_INSTALL=$PWD/opencv/install
[ "$1" == "--inside" ] && {
    shift
    export LD_LIBRARY_PATH=${OPENCV_INSTALL}/lib:${LD_LIBRARY_PATH}

    cd tkDNN/build
# prep
export TKDNN_BATCHSIZE=4
export TKDNN_MODE=FP16
if [ "$1" == '--prep' ] ; then
  echo "prepare!"
  rm yolo4tiny_*.rt || true
  ./test_yolo4tiny
  ls -l
fi
#    ./demo yolo4tiny_fp32.rt ../../video_320.mp4 y 80 4 0
#    ./demo yolo4tiny_fp32.rt ../../video_1920.mp4 y 80 4 0
#    ./demo yolo4tiny_fp16.rt ../../video_320.mp4 y 80 4 0
    ./demo yolo4tiny_fp16.rt ../../video_1920.mp4 y 80 4 0
    exit 0
}


IMAGE=andrey-task3
# docker build -t ${IMAGE} .

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
# xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

mkdir -p .cache/pip || true

docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
    --ipc=host \
    --network host \
    --security-opt apparmor:unconfined \
    --volume=$XSOCK:$XSOCK:rw \
    --volume=$XAUTH:$XAUTH:rw \
    --env="XAUTHORITY=${XAUTH}" \
    --env="DISPLAY" \
    -v $PWD/.cache/pip:/home/ubuntu/.cache/pip \
    -v $PWD:/cdir \
    -v $(readlink -f opencv):/cdir/opencv \
    -w /cdir \
    ${IMAGE} bash $0 --inside ${@:1}

