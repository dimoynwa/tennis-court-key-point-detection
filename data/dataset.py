import os
import torch
from torch.utils.data import Dataset
from .transforms import TennisKeypointsTransforms
from .read import read_data_image, read_json_data

class KeypointsDataset(Dataset):
    def __init__(self, json_data: str, 
                 image_folder: str, 
                 transfrormations: TennisKeypointsTransforms,
                 num_keypoints=14) -> None:
        super().__init__()

        assert os.path.exists(json_data)
        assert os.path.exists(image_folder)

        self.data = read_json_data(json_data)

        self.image_folder = image_folder
        self.transformations = transfrormations
        self.num_keypoints = num_keypoints

    def __getitem__(self, index):
        img_data, kps = read_data_image(self.image_folder, self.data, index)

        # CV read return image in format [h, w, c]    
        h, w = img_data.shape[:2]

        if self.transformations:
            data = self.transformations(image=img_data, keypoints=kps)
            img_data, kps = data['image'], data['keypoints']
            
            # ToTensor converts shape to [c, h, w]
            h, w = img_data.shape[1:]

        # each key point is in format (x, y). We will divide them by h and w to make them between 0 and 1
        kps = [[x / h, y / w, vis] for x, y, vis in kps]
        
        # Some examples do NOT have exact number of keypoints
        if len(kps) != self.num_keypoints:
            while len(kps) != self.num_keypoints:
                if len(kps) > self.num_keypoints:
                    kps.pop()
                else:
                    kps.append([0, 0, 0])
        
        for k in kps:
            assert k[0] >= 0 and k[0] <= 1 and k[1] >= 0 and k[1] <= 1 and k[2] in [0, 1]

        return img_data, torch.tensor(kps, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.data)