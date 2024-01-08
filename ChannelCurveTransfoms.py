import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
import cv2
from operator import eq

whitening_effect_curve_effects = {  # https://github.com/hejiann/beautify/blob/master/skin-whitening-effect.c
    "WHITENING_EFFECT_LITTLE_WHITENING":
        np.array([
            0.000000 * 255, 0.007843 * 255,
            0.121569 * 255, 0.160784 * 255,
            0.247059 * 255, 0.317647 * 255,
            0.372549 * 255, 0.462745 * 255,
            0.498039 * 255, 0.592157 * 255,
            0.623529 * 255, 0.713725 * 255,
            0.749020 * 255, 0.819608 * 255,
            0.874510 * 255, 0.913725 * 255,
            1.000000 * 255, 0.996078 * 255,
        ]).reshape(-1, 2),
    "WHITENING_EFFECT_MODERATE_WHITENING":
        np.array([
            0.000000 * 255, 0.007843 * 255,
            0.121569 * 255, 0.192157 * 255,
            0.247059 * 255, 0.372549 * 255,
            0.372549 * 255, 0.529412 * 255,
            0.498039 * 255, 0.666667 * 255,
            0.623529 * 255, 0.784314 * 255,
            0.749020 * 255, 0.874510 * 255,
            0.874510 * 255, 0.945098 * 255,
            1.000000 * 255, 0.996078 * 255,
        ]).reshape(-1, 2),
    "WHITENING_EFFECT_HIGH_WHITENING":
        np.array([
            0.000000 * 255, 0.007843 * 255,
            0.121569 * 255, 0.223529 * 255,
            0.247059 * 255, 0.427451 * 255,
            0.372549 * 255, 0.600000 * 255,
            0.498039 * 255, 0.741176 * 255,
            0.623529 * 255, 0.854902 * 255,
            0.749020 * 255, 0.933333 * 255,
            0.874510 * 255, 0.980392 * 255,
            1.000000 * 255, 0.996078 * 255,
        ]).reshape(-1, 2),
}

darkening_effect_curve_effects = {
    "DARKENING_EFFECT_LITTLE_DARKENING":
        whitening_effect_curve_effects["WHITENING_EFFECT_LITTLE_WHITENING"][:, [1, 0]],
    "WHITENING_EFFECT_MODERATE_WHITENING":
        whitening_effect_curve_effects["WHITENING_EFFECT_MODERATE_WHITENING"][:, [1, 0]],
    "WHITENING_EFFECT_HIGH_WHITENING":
        whitening_effect_curve_effects["WHITENING_EFFECT_HIGH_WHITENING"][:, [1, 0]],
}

curves = {
    "+3": whitening_effect_curve_effects["WHITENING_EFFECT_HIGH_WHITENING"],
    "+2": whitening_effect_curve_effects["WHITENING_EFFECT_MODERATE_WHITENING"],
    "+1": whitening_effect_curve_effects["WHITENING_EFFECT_LITTLE_WHITENING"],
    "-1": darkening_effect_curve_effects["DARKENING_EFFECT_LITTLE_DARKENING"],
    "-2": darkening_effect_curve_effects["WHITENING_EFFECT_MODERATE_WHITENING"],
    "-3": darkening_effect_curve_effects["WHITENING_EFFECT_HIGH_WHITENING"],
}

class CHCurveTransforms(object):
    def __init__(self, curve_points: np.ndarray):
        self.curve_points = curve_points
        self.x = self.curve_points[:, 0]
        self.y = self.curve_points[:, 1]
        self.cs = CubicSpline(self.x, self.y)
        self.xs = np.arange(0, 256, 1)
        self.ys = self.cs(self.xs)

    def plot_curve(self, save_path: str = "curve.png"):
        plt.clf()
        plt.plot(self.x, self.y, "o", label="data")
        plt.plot(self.xs, self.ys, label="cubic spline")
        plt.savefig(save_path)

    def to_lookup_table(self):
        # export for cv2.LUT()
        return self.ys.astype(np.uint8)


class RGBTransforms(object):
    def __init__(self, mode: str = "LITTLE"):
        self.mode = mode
        if mode == "+1":
            self.ts = [whitening_effect_curve_effects["WHITENING_EFFECT_LITTLE_WHITENING"],
                       whitening_effect_curve_effects["WHITENING_EFFECT_LITTLE_WHITENING"],
                       whitening_effect_curve_effects["WHITENING_EFFECT_LITTLE_WHITENING"]]
        elif mode == "+2":
            self.ts = [whitening_effect_curve_effects["WHITENING_EFFECT_MODERATE_WHITENING"],
                       whitening_effect_curve_effects["WHITENING_EFFECT_MODERATE_WHITENING"],
                       whitening_effect_curve_effects["WHITENING_EFFECT_MODERATE_WHITENING"]]
        elif mode == "+3":
            self.ts = [whitening_effect_curve_effects["WHITENING_EFFECT_HIGH_WHITENING"],
                       whitening_effect_curve_effects["WHITENING_EFFECT_HIGH_WHITENING"],
                       whitening_effect_curve_effects["WHITENING_EFFECT_HIGH_WHITENING"]]
        elif mode == "-1":
            self.ts = [darkening_effect_curve_effects["DARKENING_EFFECT_LITTLE_DARKENING"],
                       darkening_effect_curve_effects["DARKENING_EFFECT_LITTLE_DARKENING"],
                       darkening_effect_curve_effects["DARKENING_EFFECT_LITTLE_DARKENING"]]
        elif mode == "-2":
            self.ts = [darkening_effect_curve_effects["WHITENING_EFFECT_MODERATE_WHITENING"],
                       darkening_effect_curve_effects["WHITENING_EFFECT_MODERATE_WHITENING"],
                       darkening_effect_curve_effects["WHITENING_EFFECT_MODERATE_WHITENING"]]
        elif mode == "-3":
            self.ts = [darkening_effect_curve_effects["WHITENING_EFFECT_HIGH_WHITENING"],
                       darkening_effect_curve_effects["WHITENING_EFFECT_HIGH_WHITENING"],
                       darkening_effect_curve_effects["WHITENING_EFFECT_HIGH_WHITENING"]]
        else:
            raise ValueError("mode must be LITTLE, MODERATE or HIGH")

    def process(self, image: np.ndarray):
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("image must be RGB")
        b, g, r = cv2.split(image)
        channels = [b, g, r]
        for i in range(3):
            channels[i] = cv2.LUT(channels[i], CHCurveTransforms(self.ts[i]).to_lookup_table())
        return cv2.merge(channels)


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
    im_ori = cv2.imread("./data/shiroi16_fm00017550__0.jpg")
    # modes = ["LITTLE", "MODERATE", "HIGH"]
    modes = ["+3", "+2", "+1", "-1", "-2", "-3"]
    for mode in modes:
        # cct0 = CHCurveTransforms(whitening_effect_curve_effect_list[f"WHITENING_EFFECT_{mode}_WHITENING"])
        cct0 = CHCurveTransforms(curves[mode])
        cct0.plot_curve(f"curve_{mode}.png")
        # save lookup table
        np.savetxt(f"lookup_table_{mode}.csv", cct0.to_lookup_table(), delimiter=",", fmt="%d")

    bgr_list, yuv_list = [], []
    for mode in modes:
        wt = RGBTransforms(mode)
        res = wt.process(im_ori)
        cv2.imwrite(f"face_{mode}.png", res)
        bgr_list.append(res)

        lut = np.loadtxt(f"lookup_table_{mode}.csv", delimiter=",", dtype=np.uint8)
        yuv_transform = YUVBrightTransforms(lut)
        res = yuv_transform.process(im_ori)
        cv2.imwrite(f"yuv_bright_{mode}.jpg", res)
        yuv_list.append(res)

    def save_results(ims, name):
        plt.clf()
        rows, cols = 2, 4
        fig, axes = plt.subplots(rows, cols)
        axes_flat = axes.flatten()
        for i, im in enumerate(ims):
            ax = axes_flat[i]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{name}.png")

    bgr_list.insert(3, im_ori)
    save_results(bgr_list, "bgr_modify_bgr")
    yuv_list.insert(3, im_ori)
    save_results(yuv_list, "yuv_modify_y")
