# https://www.pyimagesearch.com/2020/02/10/opencv-dnn-with-nvidia-gpus-1549-faster-yolo-ssd-and-mask-r-cnn/
set -e
OPENCV_INSTALL=$PWD/opencv/install

[ "$1" == "--inside" ] && {
    cd tkDNN/build
    # ./test_yolo3
    ./demo yolo3_fp32.rt ../demo/yolo_test.mp4 y
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
    --device=/dev/video0:/dev/video0 \
    -v $PWD:/cdir \
    -w /cdir \
    ${IMAGE} bash $0 --inside

