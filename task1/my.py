import PyNvCodec as nvc
import numpy as np
import os, sys
import cv2
import argparse
from h26x_extractor.h26x_parser import H26xParser
from imutils.video import FPS
import time

class Main:
    def __init__(self, args):
        self.args = args
        self.terminating = False

    def decode(self, encFile):
        nvDec = nvc.PyNvDecoder(encFile, self.args.gpuID)
        self.nvDec = nvDec
        width, height = self.get_dims()
        while not self.terminating:
            raw_surf = nvDec.DecodeSingleSurface()
            if (raw_surf.Empty()):
                print('No more video frames')
                break
            yield raw_surf

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
        nvEnc = nvc.PyNvEncoder({'preset': 'hq', 'codec': 'h264', 's': res}, self.args.gpuID)

        for cvtSurface in en:
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
        fps = FPS()
        fps.start()
        for e in en:
            fps.update()
            if i % 100 == 0:
                fps.stop()
                print(f"fps={fps.fps():3.2f}")
            yield e

    def write_bitstream(self, en):
        with open(self.args.out_file, 'wb') as f:
            for e in en:
                f.write(e)
                yield e

    def main(self):
        en = self.decode(self.args.input)
        en = iter(en)
        next(en) # initialize decoder to get image properties

        if self.args.out_file is not None or self.encode:
            en = self.encode(en)
            en = self.do_fps(en)
            if self.args.out_file is not None:
                en = self.write_bitstream(en)

        elif self.args.out is not None:
            en = self.convert(en, self.nvDec.Format(), nvc.PixelFormat.RGB)
            en = self.download(en, nvc.PixelFormat.RGB)
            en = self.do_fps(en)
            en = self.write_jpeg(en)

        else:
            raise Exception("unexpected args combination")

        en = iter(en)
        while True:
            next(en)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="input video")
    parser.add_argument('--encode', action='store_true', help="run encoder")
    parser.add_argument('--out_file', help="write bitstream to file")
    parser.add_argument('--out', help="output folder")
    parser.add_argument('--gpuid', type=int, default=0, dest='gpuID')
    args = parser.parse_args()
    if args.out_file is not None and args.out is not None:
        raise Exception("specify either --out or --out_file")
    Main(args).main()
