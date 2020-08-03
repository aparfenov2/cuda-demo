# https://www.pyimagesearch.com/2020/02/10/opencv-dnn-with-nvidia-gpus-1549-faster-yolo-ssd-and-mask-r-cnn/
set -e
OPENCV_INSTALL=$PWD/opencv/install

[ "$1" == "--inside" ] && {
    export LD_LIBRARY_PATH=${OPENCV_INSTALL}/lib:${LD_LIBRARY_PATH}
    cd build
     ./example-app --model ../yolov5s.torchscript.pt --video ../video_320.mp4
    exit 0
}


IMAGE=andrey-task4
# docker build -t ${IMAGE} .

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
# xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
    --ipc=host \
    --network host \
    --security-opt apparmor:unconfined \
    --volume=$XSOCK:$XSOCK:rw \
    --volume=$XAUTH:$XAUTH:rw \
    --env="XAUTHORITY=${XAUTH}" \
    --env="DISPLAY" \
    -v $PWD:/cdir \
    -w /cdir \
    -v $(readlink -f opencv):/cdir/opencv \
    -v $(readlink -f video_320.mp4):/cdir/video_320.mp4 \
    ${IMAGE} bash $0 --inside

