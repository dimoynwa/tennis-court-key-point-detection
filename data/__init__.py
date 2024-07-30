import torch
import tqdm
from .key_point_record import Record  # noqa: F401
from .read import read_yaml_data, read_json_data  # noqa: F401
from .dataset import KeypointsDataset
from .data_module import TennisKeypointsDataModule  # noqa: F401

def validate_dataset(ds: KeypointsDataset,
                     split: str = None, # Train, Validation or Test
                     expected_img_shape: torch.Size = torch.Size([3, 480, 480]),
                     expected_labels_shape: torch.Size = torch.Size([14, 3])):
    valid = True
    for idx in tqdm.tqdm(range(len(ds)), desc=f'Validating dataset {split}'):
        record = ds[idx]
        img, labels = record

        if img.dtype != torch.float32:
            print(f'Image dtype does NOT match at index {idx}. Wanted torch.float32,\
                  but was {img.dtype}')
            valid = False

        if img.dtype != torch.float32:
            print(f'Labels dtype does NOT match at index {idx}. Wanted torch.float32,\
                  but was {labels.dtype}')
            valid = False

        if img.shape != expected_img_shape:
            print(f'Image shape does NOT match at index {idx}.\
                   Wanted {expected_img_shape}, but was {img.shape}')
            valid = False

        if labels.shape != expected_labels_shape:
            print(f'Labels shape does NOT match at index {idx}.\
                  Wanted {expected_labels_shape}, but was {labels.shape}')
            valid = False

    return valid