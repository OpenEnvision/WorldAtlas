from typing import List
import torch
import numpy as np
import cv2
import argparse
from lietorch import SE3
from tqdm import tqdm
from droid import Droid

from worldarena.benchmark.camera_alignment import score_camera_trajectories
from worldarena.benchmark.official_backends.base import (
    BaseMetric,
    official_checkpoint_path,
)


def image_stream(image_list, stride, calib):
    """image generator"""

    K = np.eye(3)

    image_list = image_list[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(imfile)

        h0, w0, _ = image.shape
        if calib is None:
            fx, fy, cx, cy = float(w0), float(h0), float(w0) / 2.0, float(h0) / 2.0
        else:
            fx, fy, cx, cy = calib
        K[0, 0] = fx
        K[0, 2] = cx
        K[1, 1] = fy
        K[1, 2] = cy

        h1 = int(h0 * np.sqrt((512 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((512 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0

        yield t, image[None], intrinsics


def get_cameras_accuracy(pred_Rs, gt_Rs, pred_ts, gt_ts):
    """Align predicted poses to GT with Sim(3) and return camera errors."""
    pred_count = int(pred_Rs.shape[0])
    gt_count = int(gt_Rs.shape[0])
    cameras_pred = np.repeat(np.eye(4, dtype=np.float64)[None], pred_count, axis=0)
    cameras_gt = np.repeat(np.eye(4, dtype=np.float64)[None], gt_count, axis=0)
    cameras_pred[:, :3, :3] = pred_Rs.detach().cpu().numpy()
    cameras_pred[:, :3, 3] = pred_ts.detach().cpu().numpy()
    cameras_gt[:, :3, :3] = gt_Rs.detach().cpu().numpy()
    cameras_gt[:, :3, 3] = gt_ts.detach().cpu().numpy()
    (r_score, t_score), _ = score_camera_trajectories(cameras_pred, cameras_gt)
    print(f"Rotation error: {r_score}")
    print(f"Translation error: {t_score}")
    return (r_score, t_score)


class CameraErrorMetric(BaseMetric):
    """

    return: (R_score, T_score)

    Range: [0, 1] higher the better
    """

    def __init__(self) -> None:
        super().__init__()
        args = {
            "t0": 0,
            "stride": 1,
            "weights": official_checkpoint_path("droid.pth"),
            "buffer": 512,
            "beta": 0.3,
            "filter_thresh": 0.01,
            "warmup": 8,
            "keyframe_thresh": 4.0,
            "frontend_thresh": 16.0,
            "frontend_window": 25,
            "frontend_radius": 2,
            "frontend_nms": 1,
            "backend_thresh": 22.0,
            "backend_radius": 2,
            "backend_nms": 3,
            # need high resolution depths
            "upsample": True,
            "stereo": False,
            "calib": None,
        }
        args = argparse.Namespace(**args)

        self._args = args
        self.droid = None
        try:
            torch.multiprocessing.set_start_method("spawn")
        except Exception as e:
            print(f"Error setting start method: {e}")

    def _compute_scores(
        self, rendered_images: List[str], cameras_gt: torch.Tensor, scale: float = 1.0
    ) -> float:

        for t, image, intrinsics in tqdm(
            image_stream(rendered_images, self._args.stride, self._args.calib)
        ):
            if t < self._args.t0:
                continue

            if self.droid is None:
                self._args.image_size = [image.shape[2], image.shape[3]]
                self.droid = Droid(self._args)
            self.droid.track(t, image, intrinsics=intrinsics)

        traj_est, _ = self.droid.terminate(
            image_stream(rendered_images, self._args.stride, self._args.calib)
        )
        cameras_pred = SE3(torch.as_tensor(traj_est)).matrix()

        to_blender = torch.diag(
            torch.tensor([-1.0, 1.0, 1.0, 1.0], device=cameras_pred.device)
        )
        cameras_pred = to_blender @ cameras_pred

        to_blender = torch.diag(
            torch.tensor([1.0, 1.0, -1.0, 1.0], device=cameras_pred.device)
        )
        w2c_cameras_pred = torch.inverse(cameras_pred)
        w2c_cameras_pred = to_blender @ w2c_cameras_pred
        cameras_pred = torch.inverse(w2c_cameras_pred)

        camera_error, alignment_details = score_camera_trajectories(
            cameras_pred.detach().cpu().numpy(),
            cameras_gt.detach().cpu().numpy(),
            gt_scale=float(scale),
        )
        self._alignment_details = alignment_details

        self.droid = None
        return camera_error
