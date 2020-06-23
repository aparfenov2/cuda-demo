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

    def main(self):
        os.makedirs(self.args.out, exist_ok=True)
        en = self.decode(self.args.input)
        en = iter(en)
        next(en)

        en = self.encode(en)
        # en = self.convert(en, self.nvDec.Format(), nvc.PixelFormat.RGB)
        # en = self.download(en, nvc.PixelFormat.RGB)

        fps = FPS()
        fps.start()

        # H26xParser.set_callback("nalu", do_something)
        # H26xParser.parse()        

        for i, e in enumerate(en):
            # print(i, e.shape)
            fps.update()
            if i % 100 == 0:
                print(f"fps={fps.fps():3.2f}")
            # cv2.imwrite(f'{self.args.out}/{i:03}.jpg', e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="input video")
    parser.add_argument('--out', default='out', help="output folder")
    parser.add_argument('--gpuid', type=int, default=0, dest='gpuID')
    args = parser.parse_args()
    Main(args).main()
