import numpy as np


# ------------------------------------------------------------------
# New:  interpolate_bboxes
# ------------------------------------------------------------------
# Works for ANY number of detection points (N ≥ 1).
# Between every consecutive pair (i, i+1) it inserts
# `num_intermediate` linearly-spaced ghost boxes.
#
# Why this beats the old YOLO-based fallback:
#   • No model inference  → ~0 extra CPU / GPU cost
#   • Works for N = 1, 2, 3, …  (not just the 2-point edge-case)
#   • Pure numpy  → vectorised, sub-millisecond
#
# bbox format:  [(x1, y1), (x2, y2)]   (top-left / bottom-right)
# ------------------------------------------------------------------

def interpolate_bboxes(bbox_list, num_intermediate=2):
    """
    Densify a list of N bounding boxes by linearly interpolating
    `num_intermediate` extra boxes between every consecutive pair.

    Args:
        bbox_list       : list of [(x1,y1),(x2,y2)] in full-frame pixel coords
        num_intermediate: how many new boxes to insert between each adjacent pair
                          0  → returns bbox_list unchanged (just originals)
                          2  → inserts 2 ghost boxes, so 3 detections become 7

    Returns:
        Densified list of [(x1,y1),(x2,y2)] — originals preserved in-order.
    """
    n = len(bbox_list)
    if n <= 1 or num_intermediate <= 0:
        return list(bbox_list)  # nothing to interpolate

    # Convert to a (N, 4) float32 array  [x1, y1, x2, y2]
    arr = np.array(
        [[b[0][0], b[0][1], b[1][0], b[1][1]] for b in bbox_list],
        dtype=np.float32
    )

    result = []
    for i in range(n - 1):
        start = arr[i]      # shape (4,)
        end   = arr[i + 1]  # shape (4,)

        # Append the anchor box
        result.append(bbox_list[i])

        # Generate `num_intermediate` evenly-spaced boxes (excluding endpoints)
        # np.linspace(0,1, num_intermediate+2)[1:-1]  gives the interior t-values
        for t in np.linspace(0.0, 1.0, num_intermediate + 2)[1:-1]:
            interp = start + t * (end - start)
            x1, y1, x2, y2 = int(interp[0]), int(interp[1]), int(interp[2]), int(interp[3])
            result.append([(x1, y1), (x2, y2)])

    # Append the final anchor
    result.append(bbox_list[-1])
    return result


# ------------------------------------------------------------------
# Legacy helpers — kept for any external callers
# ------------------------------------------------------------------

def list_of_points(pt1, pt2, length):
    a = (pt2[0] - pt1[0]) / length
    b = (pt2[1] - pt1[1]) / length
    ab = [pt1]
    at = pt1[0]
    bt = pt1[1]
    for _ in range(length - 2):
        at = at + a
        bt = bt + b
        ab.append((round(at), round(bt)))
    ab.append(pt2)
    return ab


def rect_points(list1, length):
    if len(list1) == 1:
        return list1

    rect_list = list()
    for i in range(0, len(list1) - 1):
        start = list_of_points(list1[i][0], list1[i + 1][0], length)
        end   = list_of_points(list1[i][1], list1[i + 1][1], length)
        for j, k in zip(start, end):
            rect_list.append([(int(j[0]), int(j[1])), (int(k[0]), int(k[1]))])
    return rect_list
