import PyNvCodec as nvc
import numpy as np
import os, sys
import cv2
import argparse
from bitstring import BitStream
from itertools import islice
# from imutils.video import FPS
import time
from collections import namedtuple

ParserResult = namedtuple("ParserResult",['typ','bytes'])

class FPS:
    def __init__(self):
        self.start()

    def start(self):
        self._start = time.time()
        self.frames = 0

    def stop(self):
        pass

    def update(self):
        self.frames += 1
        _end   = time.time()

    def fps(self):
        _end   = time.time()
        return  self.frames / (_end - self._start)

class FPSCounter:
    def __init__(self, avg_len=5):
        self.avg_len = avg_len
        self.fps = FPS()
        self.fps.start()
        self.fps.stop()
        self.last_fps = []

    def update(self):
        self.fps.update()

    def get_last_fps(self):
        return sum(self.last_fps)/len(self.last_fps)

    def fps_job(self):
        self.fps.stop()
        self.last_fps.append(self.fps.fps())
        self.fps.start()
        if len(self.last_fps) > self.avg_len:
            self.last_fps.pop(0)
        print(f"fps: curr={self.last_fps[-1]:3.2f}, min={min(self.last_fps):3.2f}, avg={self.get_last_fps():3.2f}, max={max(self.last_fps):3.2f}")

class NaluSizeMeter:
    def __init__(self, avg_len=5):
        self.avg_len = avg_len
        self.last_size = []

    def update(self,sz):
        self.last_size.append(sz)
        if len(self.last_size) > self.avg_len:
            self.last_size.pop(0)

    def print_rarely(self):
        print(f"nalu_size: curr={self.last_size[-1]:3.2f}, min={min(self.last_size):3.2f}, avg={sum(self.last_size)/len(self.last_size):3.2f}, max={max(self.last_size):3.2f}")

class Main:
    def __init__(self, args):
        self.args = args
        self.terminating = False

    def decode(self, encFile):
        nvDec = nvc.PyNvDecoder(encFile, self.args.gpuID)
        self.nvDec = nvDec
        width, height = self.get_dims()
        i = 0
        while not self.terminating:
            if self.args.limit is not None:
                if i > self.args.limit:
                    print(f'limit of {self.args.limit} frames reached. Stream terminated.')
                    break
            raw_surf = nvDec.DecodeSingleSurface()
            if (raw_surf.Empty()):
                print('No more video frames')
                break
            yield raw_surf
            i += 1

    def get_dims(self):
        width, height = self.nvDec.Width(), self.nvDec.Height()
        return width, height

    def convert(self, en, _from, _to):
        width, height = self.get_dims()
        to_rgb = nvc.PySurfaceConverter(width, height, _from, _to, self.args.gpuID) # nvc.PixelFormat.YUV420
        for raw_surf in en:
            rgb_surf = to_rgb.Execute(raw_surf)
            if (rgb_surf.Empty()):
                print(f'convert from {_from} to {_to} failed')
                break
            yield rgb_surf

    def download(self, en, _format):
        width, height = self.get_dims()
        nvDwn = nvc.PySurfaceDownloader(width, height, _format, self.args.gpuID)
        for rgbp_surf in en:
            rawFrame = np.ndarray(shape=(rgbp_surf.HostSize()), dtype=np.uint8)
            success = nvDwn.DownloadSingleSurface(rgbp_surf, rawFrame)
            if not (success):
                print('Failed to download surface')
                break
            yield rawFrame.reshape((height, width, 3))

    def encode(self, en):
        width, height = self.get_dims()
        res = f"{width}x{height}"
        conf = dict(args.conf)
        if 'preset' not in conf:
            conf['preset'] = 'hq'
        if 'codec' not in conf:
            conf['codec'] = 'h264'
        if 's' not in conf:
            conf['s'] = res

        print("effective encocer params:", conf)
        nvEnc = nvc.PyNvEncoder(conf, self.args.gpuID)

        for cvtSurface in en:
            # if not self.args.no_force_idr:
            #     nvEnc.Reconfigure({}, force_idr=True)
            encFrame = np.ndarray(shape=(0), dtype=np.uint8)
            success = nvEnc.EncodeSingleSurface(cvtSurface, encFrame)
            if not success:
                print("WARN: encode failed")
                time.sleep(0.0001)
                continue
            bits = bytearray(encFrame)
            yield bits

    def write_jpeg(self, en):
        os.makedirs(self.args.out, exist_ok=True)
        for i, e in enumerate(en):
            cv2.imwrite(f'{self.args.out}/{i:03}.jpg', e)
            yield e

    def do_fps(self, en):
        fps = FPSCounter(avg_len=self.args.avg_len)
        for i,e in enumerate(en):
            fps.update()
            if i % 100 == 0:
                fps.fps_job()
            yield e

    def do_nalu_size(self, en):
        sz = NaluSizeMeter(avg_len=self.args.avg_nalu_len)
        for i,e in enumerate(en):
            sz.update(len(e.bytes))
            if i % 100 == 0:
                sz.print_rarely()
            yield e

    def do_nalu_size2(self, en):
        _start = time.time()
        _sum = 0
        for i,e in enumerate(en):
            _sum += len(e.bytes)
            if i % 1000 == 0:
                _end = time.time()
                print(f'cummulative size:{_sum} bytes, bitrate: {_sum/(_end-_start):5.2f}, bytes')
                _start = time.time()
                _sum = 0
            yield e

    def parse_bitstream(self, en):
        START_CODE_PREFIX = b'\x00\x00\x00\x01'

        _iter = iter(en)

        # look for 1st NALU, skip anything before
        chunk = next(_iter)
        offset = 0
        first_nalu_pos = chunk.find(START_CODE_PREFIX, offset)
        while first_nalu_pos < 0:
            offset = max(offset, len(chunk) - len(START_CODE_PREFIX))
            chunk += next(_iter)
            first_nalu_pos = chunk.find(START_CODE_PREFIX, offset)

        while True:
            # read until next NALU or end of stream
            offset = first_nalu_pos + len(START_CODE_PREFIX)
            next_nalu_pos = chunk.find(START_CODE_PREFIX, offset)
            while next_nalu_pos < 0:
                offset = max(offset, len(chunk) - len(START_CODE_PREFIX))
                chunk += next(_iter)
                next_nalu_pos = chunk.find(START_CODE_PREFIX, offset)

            yield chunk[first_nalu_pos:next_nalu_pos]
            first_nalu_pos = next_nalu_pos
            # remove old data
            chunk = chunk[first_nalu_pos:]
            first_nalu_pos = 0

    def _decode_nalu(self, nalu_bytes):
        """
        Returns nal_unit_type and RBSP payload from a NALU stream
        """
        START_CODE_PREFIX = "0x00000001"
        if "0x" + nalu_bytes[0: 4*8].hex == START_CODE_PREFIX:
            start_code = nalu_bytes.read('bytes:4')
        else:
            start_code = nalu_bytes.read('bytes:3')
        forbidden_zero_bit = nalu_bytes.read(1)
        nal_ref_idc = nalu_bytes.read('uint:2')
        nal_unit_type = nalu_bytes.read('uint:5')

        return nal_unit_type


    def decode_nalu(self, en):
        NAL_UNIT_TYPE_SPS = 7    # Sequence parameter set
        NAL_UNIT_TYPE_PPS = 8    # Picture parameter set
        for e in en:
            typ = 'slice'
            nal_unit_type = self._decode_nalu(BitStream(e))
            if nal_unit_type == NAL_UNIT_TYPE_SPS:
                typ = 'sps'
            elif nal_unit_type == NAL_UNIT_TYPE_PPS:
                typ = 'pps'
            # print(f'{nal_unit_type}:{typ}')
            yield ParserResult(typ, e)

    def write_bin(self, en):
        os.makedirs(self.args.out, exist_ok=True)
        hdr = []
        START_CODE_PREFIX = b'\x00\x00\x00\x01'
        for i, e in enumerate(en):
            if e.typ in ['sps','pps']:
                print(f'{e.typ} received')
                hdr.append(e)
                yield e
                continue
            outfile = f'{self.args.out}/{i:03}.h264'
            if i % 10 == 0:
                print(outfile)
            with open(outfile, 'wb') as f:
                for h in hdr:
                    f.write(h.bytes)
                f.write(e.bytes)
                f.write(START_CODE_PREFIX)
            yield e

    def write_bitstream(self, en):
        with open(self.args.single_file, 'wb') as f:
            for e in en:
                f.write(e)
                yield e

    def decode_with_loop(self):
        while True:
            print(f"start decoding file {self.args.input}")
            en = self.decode(self.args.input)
            _iter = iter(en)
            while True:
                try:
                    yield next(_iter)
                except StopIteration:
                    break
            if not self.args.loop:
                break

    def nalu_to_raw(self, en):
        for e in en:
            yield e.bytes

    def main(self):
        en = self.decode_with_loop()
        en = iter(en)
        next(en) # initialize decoder to get image properties

        en = self.do_fps(en)

        if self.args.encode:
            print('adding encoder to pipeline')
            en = self.encode(en)

        if self.args.parse:
            assert self.args.encode, "encoder should be enabled for parser"

            print('adding parser to pipeline')
            en = self.parse_bitstream(en)
            en = self.decode_nalu(en)
            if self.args.print_size:
                en = self.do_nalu_size2(en)

        if self.args.single_file:
            assert self.args.encode, "encoder should be enabled for single_file"
            print("writing bitstream to single file")
            if self.args.parse:
                en = self.nalu_to_raw(en)
            en = self.write_bitstream(en)

        if self.args.out:
            if self.args.encode:
                assert self.args.parse, "parser should be enabled for I frames saver"
                print("writing I frames to multiple files")
                en = self.write_bin(en)
            else:
                print("writing frames to JPEG files")
                en = self.convert(en, self.nvDec.Format(), nvc.PixelFormat.RGB)
                en = self.download(en, nvc.PixelFormat.RGB)
                en = self.write_jpeg(en)

        en = iter(en)
        while True:
            next(en)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="input video")
    parser.add_argument('--encode', action='store_true', help="run encoder only")
    parser.add_argument('--parse', action='store_true', help='run parser')
    parser.add_argument('--single_file', help="write bitstream to file")
    parser.add_argument('--out', help="output folder")
    parser.add_argument('--gpuid', type=int, default=0, dest='gpuID')
    parser.add_argument('--loop', action='store_true', help='repeat reading file')
    parser.add_argument('--limit', type=int, help='limit nmuber of frames')
    parser.add_argument('--avg_len', type=int, default=5, help='average fps len')
    parser.add_argument('--avg_nalu_len', type=int, default=5, help='average nalu size acc len')
    parser.add_argument('--print_size', action='store_true', help='print avg nalu size')
    parser.add_argument('--conf', nargs=2, action='append')

    args = parser.parse_args()
    print("cmdline encoder params: ", args.conf)

    if args.parse:
        args.encode = True

    Main(args).main()
