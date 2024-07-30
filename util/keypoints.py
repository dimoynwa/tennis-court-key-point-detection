import numpy as np

def order_points(kps: list[tuple|list], return_center=False, alpha=10):
    """
    Processes and sorts an array of arrays of 2 elements [x, y].

    This function performs the following steps:
    1. Takes an input array of arrays, each containing two elements [x, y].
    2. Sorts the input array by the second element (y) where x and y are not 0.
    3. While the new array is not empty:
        3.1. Calculates the minimum y coordinate from the remaining points.(Just get the first point y coordinate
            as they are already sorted)
        3.2. Pops the first element from the new array while the difference 
             between the element's y and the minimum y is less than or equal to alpha.
        3.3. Adds the popped elements to a new array (batch).
        3.4. Sorts the popped elements (batch) by the first element (x) and adds them to the result.
        3.4. Empty the batch
        3.5. Repeats the process (3.) until the new array is empty.
    4. Adds all element with x == y == 0 from the original array at the end of result array.
        
    Args:
        kps (list[tuple | list]): keypoints. Can be points of type [x, y] or [x, y, vis] where vis is visibility(0 or 1)
        return_center (bool, optional): if True the center point will be returned. Defaults to False.
        alpha (int): The possible difference between 2 points y coordinate to be in a batch. Defaults to 10.
    Returns:
        list: Keypoints sorted by the distance to the center.
    """
    non_zero_kps = [kp[:2] for kp in kps if kp[0] != 0 and kp[1] != 0]
    zero_kps = [kp[:2] for kp in kps if kp[0] == kp[1] == 0]
    # sort by y coordinate
    sorted_kps = sorted(non_zero_kps, key=lambda point: point[1])
    new_kps = []

    min_y = non_zero_kps[0][1]

    while len(sorted_kps) > 0:

        batch = []
        while len(sorted_kps) > 0 and sorted_kps[0][1] - min_y <= alpha:
            # Add first point to the batch
            batch.append(sorted_kps.pop(0))
        
        if len(batch) > 0:
            # Append points to the result sorted by x coordinate
            new_kps.extend(sorted(batch, key=lambda point: point[0]))
            batch = []

        if len(sorted_kps) > 0:
            min_y = sorted_kps[0][1]

    new_kps.extend(zero_kps)

    if return_center:
        center_point =  np.array(non_zero_kps).mean(axis=0).tolist()
        return new_kps, center_point
    return new_kps