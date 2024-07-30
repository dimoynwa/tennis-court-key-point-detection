import torch
import numpy as np
import cv2
from util.unnormalize import UnNormalize

def draw_tensor(img_data: torch.Tensor, keypoints: torch.Tensor, conf_threshold=.5,
                unnormalize = True,
                **kwargs):
    kps = keypoints[keypoints[:, 2] > conf_threshold]

    if unnormalize:
        unnorm = UnNormalize()
        img_data = unnorm(img_data) # (3, 480, 480)
    
    img = img_data.permute(1, 2, 0).numpy() # (480, 480, 3)
    img = img * 255 

    img = np.ascontiguousarray(img, dtype=np.uint8)

    keypoints_t = [[(int(x)), int(y)] for x, y, _ in kps]
    img = draw_keypoints(img, keypoints_t, thickness=2, **kwargs)

    return img

def draw_keypoints(frame, keypoints, color=(255, 0, 0), thickness=4):

    for i, (x, y) in enumerate(keypoints):
        frame = cv2.circle(frame, center=(int(x), int(y)), radius=4, color=color, thickness=thickness)
        frame = cv2.putText(frame, f'#{i}', (int(x), int(y) - 10), cv2.FONT_HERSHEY_COMPLEX, .8, color, thickness)
    return frame