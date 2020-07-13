false && {
bash _run.sh --prep --mode fp32 --batch 1 --320
bash _run.sh --prep --mode fp32 --batch 1 --1920
bash _run.sh --prep --mode fp32 --batch 4 --320
bash _run.sh --prep --mode fp32 --batch 4 --1920
}
false && {
bash _run.sh --prep --mode fp16 --batch 1 --320
bash _run.sh --prep --mode fp16 --batch 1 --1920
bash _run.sh --prep --mode fp16 --batch 4 --320
bash _run.sh --prep --mode fp16 --batch 4 --1920
}

true && {
bash _run.sh --prep --mode int8 --batch 1 --320
bash _run.sh --prep --mode int8 --batch 1 --1920
bash _run.sh --prep --mode int8 --batch 4 --320
bash _run.sh --prep --mode int8 --batch 4 --1920
}
