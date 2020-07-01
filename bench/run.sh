docker run --gpus all -it \
    -v $PWD:/cdir \
    -w /cdir \
    tensorflow/tensorflow:latest-gpu bash -c "pip install ai-benchmark && python my-benchmark.py"
