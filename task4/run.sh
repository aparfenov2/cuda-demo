
[ "$1" == "--inside" ] && {
shift
# . .env/bin/activate
# python --version
cd yolov5
# python detect.py --source ../big_buck_bunny_1080p_h264.mov
python detect.py --device 0 \
    --source /cdir/video_320.mp4 \
    --output /cdir/output \
    --save-txt
exit 0
}

docker run --gpus all --ipc=host -it --rm \
    -v $PWD:/cdir \
    -w /cdir \
    -v $(readlink -f video_320.mp4):/cdir/video_320.mp4 \
    ultralytics/yolov5:latest bash /cdir/$0 --inside $@
