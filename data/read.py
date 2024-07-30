import os
import yaml
import json
import cv2

from util.keypoints import order_points
from .key_point_record import Record
from util.constants import IMAGE_EXT

def read_yaml_data(file_path: str) -> list[Record]:
    """Reads yaml file containing list of Records and return them

    Args:
        file_path (str): path to the data file

    Returns:
        list[Record]: all records in th data set
    """
    
    assert os.path.exists(file_path)
    with open(file_path, 'r') as data_f:
        content = data_f.read()
        records_list = yaml.safe_load(content)

    return records_list

def read_json_data(file_path: str) -> list[Record]:
    """Reads JSON file containig list of records and returns them

    Args:
        file_path (str): path to the data file

    Returns:
        list[Record]: all records in th data set
    """
    assert os.path.exists(file_path)
    with open(file_path, 'r') as data_f:
        records_list = json.load(data_f)

    return records_list

def read_data_image(image_dir: str, data: list, idx: int):
    """Reads an image from image_dir by given data and given index
    and return its content as numpy.ndarray and keypoints.
    Keypoints are sorted based on order_points_v2 method.

    Args:
        image_dir (str): the directory where images are located
        data (list): data records array
        idx (int): index of wanted image

    Returns:
        tuple: tuple of image_data (numpy.ndarray) and keypoints (list with shape (14, 2))
    """
    record = data[idx]
    if image_dir.endswith('/'):
        image_dir = image_dir[:-1]
    file_path = f'{image_dir}/{record["id"]}{IMAGE_EXT}'
    assert os.path.exists(file_path)
    frame = cv2.imread(filename=file_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    kps = order_points(record['kps'])

    return frame, kps