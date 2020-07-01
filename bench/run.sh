[ "$1" == "--inside" ] && {
    pip install ai-benchmark
    ln -s /usr/local/lib/python3.6/dist-packages/ai-benchmark/data data
    ln -s /usr/local/lib/python3.6/dist-packages/ai-benchmark/models models
    ls -l data
    python my-benchmark.py
    exit 0
}

mkdir -p .cache/pip || true
sudo chown -R root:root .cache
docker run --gpus all -it \
    -v $PWD/.cache/pip:/root/.cache/pip \
    -v $PWD:/cdir \
    -w /cdir \
    tensorflow/tensorflow:latest-gpu bash $0 --inside
