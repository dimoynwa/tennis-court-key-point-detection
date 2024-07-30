from typing import Any, Callable
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class TennisKeypointsTransforms(Callable):
    def __init__(self, 
                 img_size: int|tuple[int],
                 rotate: int,
                 normalize:bool=True,
                 keypoints_format='xy'):
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        transformations = []
        if (rotate > 0):
            transformations.append(A.Rotate(rotate))

        transformations.append(A.Resize(self.img_size[0], self.img_size[1]))
        if normalize:
            transformations.append(
                A.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)))

        transformations.append(ToTensorV2(always_apply=True))

        self.transform = A.Compose(
            transformations,
            keypoint_params=A.KeypointParams(format=keypoints_format, remove_invisible=False))

    def __call__(self, image: np.ndarray, keypoints: list) -> dict[str, Any]:
        # image will be MatLike of shape [W, H, C]
        w, h = image.shape[:2]

        # Add visibility parameter to points
        keypoints_filtered = [self.create_new_keypoint(kp, self.is_point_visible(kp, (w, h)))\
                              for kp in keypoints]

        assert len(keypoints) == len(keypoints_filtered)

        # Get points for transformation
        kps = keypoints if len(keypoints[0]) == 2 else [kp[:2] for kp in keypoints_filtered]

        # Transforms
        data = self.transform(image=image, keypoints=kps)
        trn_image, trn_kps = data['image'], data['keypoints']

        assert len(keypoints) == len(trn_kps)

        ## Add previous visiblity calculated
        trn_kps = [[kp[0], kp[1], keypoints_filtered[idx][2]] for idx, kp in enumerate(trn_kps)]

        # Every element of trn_kps is an array of 2 elements [x, y] 
        # Let's add visibility to the point

        trn_kps = [self.create_new_keypoint(kp, self.is_point_visible(kp, self.img_size)) for kp in trn_kps]
        
        # Push not visible at the end
        trn_kps = sorted(trn_kps, key=lambda x: x[2], reverse=True)

        return {
            'image': trn_image,
            'keypoints': trn_kps
        }            

    def is_point_visible(self, point: tuple[int|float]|list[int|float], img_size: tuple[int]=None) -> bool:
        """Check if point is visible or not. If len of point is 3 and it is already invisible,
        we won't check and will return False.
        Point is visible if x, y coordinages are in range [0, width/height]

        Args:
            point (tuple[int | float]): point with x, y coordinates
            img_size (tuple[int], optional): Tuple of (width, height). Defaults to None. If None, self.img_size will be get

        Returns:
            bool: x >= 0 and x < width and y >= 0 and y < height
        """
        if len(point) == 3 and point[2] == 0:
            return False

        x, y = point[0], point[1]
        h, w = img_size if img_size else self.img_size

        return x >= 0 and x < w and y >= 0 and y < h

    def create_new_keypoint(self,
                            point: tuple[int|float]|list[int|float],
                            visible: bool) -> list[int|float]:
        """Function that accepts point and visibility state.
        Return a list with 3 values [x, y, vis], where x, y are point coordinates
        if the point is visible, 0, 0 otherwise and vis = 1 if point is visible,
        0 otherwise.

        Args:
            point (tuple[int | float]): key point
            visible (bool): if point is visible or not

        Returns:
            list[int|float]: list with format [x, y, vis]
        """
        assert len(point) in [2, 3]
        if visible:
            return [point[0], point[1], 1]
        return [0, 0, 0]