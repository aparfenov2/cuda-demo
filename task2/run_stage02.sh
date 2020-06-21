# https://www.pyimagesearch.com/2020/02/10/opencv-dnn-with-nvidia-gpus-1549-faster-yolo-ssd-and-mask-r-cnn/
set -e
OPENCV_INSTALL=$PWD/opencv/install

[ "$1" == "--inside" ] && {
    python3 -m pip install -r requirements.txt
    export LD_LIBRARY_PATH=${OPENCV_INSTALL}/lib:${LD_LIBRARY_PATH}
    export PYTHONPATH=${OPENCV_INSTALL}/lib/python3.6/dist-packages:$PYTHONPATH

    python3 ssd_object_detection.py \
        --prototxt face_ssd/deploy.prototxt.txt \
        --model face_ssd/res10_300x300_ssd_iter_140000.caffemodel \
        --input rtsp://test:test@192.168.250.110/axis-media/media.amp \
        --display 0 \
        --use-gpu 1 \
        --confidence 0.6 \
        --workers 4 \
        --fixed_size 300x300 \

        # --output ssd_guitar.avi \
        # --input example_videos/guitar.mp4 \
        # --prototxt opencv-ssd-cuda/MobileNetSSD_deploy.prototxt \
        # --model opencv-ssd-cuda/MobileNetSSD_deploy.caffemodel \

    exit 0
}


IMAGE=andrey-task2
# docker build -t ${IMAGE} .

# XSOCK=/tmp/.X11-unix
# XAUTH=/tmp/.docker.xauth
# touch $XAUTH
# xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

mkdir -p .cache/pip || true

docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
    --ipc=host \
    --network host \
    --security-opt apparmor:unconfined \
    --env="XAUTHORITY=${XAUTH}" \
    --env="DISPLAY" \
    -v $PWD/.cache/pip:/home/ubuntu/.cache/pip \
    -v $(readlink -f face_ssd):/cdir/face_ssd \
    -v $PWD:/cdir \
    -w /cdir \
    ${IMAGE} bash $0 --inside

