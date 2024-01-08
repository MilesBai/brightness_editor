import cv2
import numpy as np
from operator import eq


class YUVBrightTransforms(object):
    def __init__(self, lut: np.ndarray):
        assert eq(lut.shape, (256,))
        self.lut = lut.astype(np.uint8)

    def process(self, im: np.ndarray):
        assert eq(im.shape[2], 3)
        im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(im_yuv)
        y_ = cv2.LUT(y, self.lut)

        im_yuv_ = cv2.merge((y_, u, v))
        im_bgr = cv2.cvtColor(im_yuv_, cv2.COLOR_YUV2BGR)
        return im_bgr


if __name__ == "__main__":
    modes = ["LITTLE", "MODERATE", "HIGH"]
    yuv_transforms = []
    for mode in modes:
        lut = np.loadtxt(f"lookup_table_{mode}.csv", delimiter=",", dtype=np.uint8)
        yuv_transforms.append(YUVBrightTransforms(lut))

        res = yuv_transforms[-1].process(cv2.imread("shiroi16_fm00017550__0.jpg"))
        cv2.imwrite(f"yuv_bright_{mode}.jpg", res)
