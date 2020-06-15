import PyNvCodec as nvc
import numpy as np
import os, sys
import cv2

class Main:
    def emit_frames(self):
        gpuID = 0
#        encFile = "big_buck_bunny_1080p_h264.mov"
        encFile = sys.argv[1]

        self.nvDec = nvDec = nvc.PyNvDecoder(encFile, gpuID)
        width, height = self.nvDec.Width(), self.nvDec.Height()

        #Amount of memory in RAM we need to store decoded frame
        # frameSize = nvDec.Framesize()
        # rawFrameNV12 = np.ndarray(shape=(frameSize), dtype=np.uint8)

    # to_rgb = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID)
    # to_planar = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpuID)

# https://github.com/NVIDIA/VideoProcessingFramework/issues/36
# https://stackoverflow.com/questions/2231518/how-to-read-a-frame-from-yuv-file-in-opencv
        self.to_rgb = nvc.PySurfaceConverter(width, height, self.nvDec.Format(), nvc.PixelFormat.RGB, gpuID) # nvc.PixelFormat.YUV420
        # self.nvRes = nvc.PySurfaceResizer(hwidth, hheight, self.to_rgb.Format(), gpuID)
        self.to_planar = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpuID)
        self.nvDwn = nvc.PySurfaceDownloader(width, height, self.to_rgb.Format(), gpuID)

        while True:
            # success = nvDec.DecodeSingleFrame(rawFrameNV12)
            # if not (success):
            #     print("stream terminated")
            #     break
            raw_surf = self.nvDec.DecodeSingleSurface()
            if (raw_surf.Empty()):
                print('No more video frames')
                break

            rgb_surf = self.to_rgb.Execute(raw_surf)
            if (rgb_surf.Empty()):
                print('to_rgb failed')
                break

            rgbp_surf = self.to_planar.Execute(rgb_surf)
            if (rgbp_surf.Empty()):
                print('to_planar failed')
                break

            rawFrame = np.ndarray(shape=(rgbp_surf.HostSize()), dtype=np.uint8)
            success = self.nvDwn.DownloadSingleSurface(rgb_surf, rawFrame)
            if not (success):
                print('Failed to download surface')
                break

            yield rawFrame.reshape((height, width, 3))

    def main(self):
        os.makedirs('out', exist_ok=True)
        en = self.emit_frames()
        for i, e in enumerate(en):
            print(i, e.shape)
#            cv2.imwrite(f'out/{i:03}.jpg', e)

if __name__ == '__main__':
    Main().main()
