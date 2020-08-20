# https://www.pyimagesearch.com/2020/02/10/opencv-dnn-with-nvidia-gpus-1549-faster-yolo-ssd-and-mask-r-cnn/
set -e
OPENCV_INSTALL=$PWD/opencv/install
video=video_320.mp4
_TKDNN_MODE=fp32
_TKDNN_BATCHSIZE=1
_SHOW=0
POSITIONAL=("$@")
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --320) video="video_320.mp4" ;;
        --1920) video="video_1920.mp4" ;;
        --inside) inside=1 ;;
        --batch) _TKDNN_BATCHSIZE="$2"; shift ;;
        --mode) _TKDNN_MODE="$2"; shift ;;
        --show) _SHOW=1 ;;
        --prep) prep=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo -------------
echo POSITIONAL=${POSITIONAL[@]}
echo MODE=${_TKDNN_MODE}
echo BSIZE=${_TKDNN_BATCHSIZE}
echo video=${video}
echo prep=${prep}

[ -n "$inside" ] && {
    export LD_LIBRARY_PATH=${OPENCV_INSTALL}/lib:${LD_LIBRARY_PATH}

    cd tkDNN/build
# prep
[ "${_TKDNN_BATCHSIZE}" -gt 1 ] && {
    export TKDNN_BATCHSIZE=${_TKDNN_BATCHSIZE}
}
[ "${_TKDNN_MODE}" == "int8" ] && {
    export TKDNN_CALIB_LABEL_PATH=../demo/COCO_val2017/all_labels.txt
    export TKDNN_CALIB_IMG_PATH=../demo/COCO_val2017/all_images.txt
    [ -f "${TKDNN_CALIB_IMG_PATH}" ] || {
        echo COCO calib files not found. Please run bash scripts/download_validation.sh COCO first.
        exit 0
    }
}

export TKDNN_MODE=${_TKDNN_MODE^^}
[ -n "$prep" ] && {
  echo "prepare!"
  rm yolo4tiny_*.rt || true
  ./test_yolo4tiny
  # ls -l
}
set -x
    ./demo yolo4tiny_${_TKDNN_MODE}.rt ../../${video} y 80 ${_TKDNN_BATCHSIZE} ${_SHOW}
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
    -v $PWD:/home/amtg/demo/task3 \
    -v $(readlink -f opencv):/cdir/opencv \
    -w /cdir \
    ${IMAGE} bash $0 --inside "${POSITIONAL[@]}"

