
# x1, y1, x2, y2 -> x1 < x2, y1 < y2
def standard_bbox(bbox):
    x_coordinates = sorted(bbox[0::2])
    y_coordinates = sorted(bbox[1::2])
    bbox = [x_coordinates[0], y_coordinates[0], x_coordinates[1], y_coordinates[1]]
    return bbox

def normalize_bbox(bbox):
    return [
        int(1000 * bbox[0]),
        int(1000 * bbox[1]),
        int(1000 * bbox[2]),
        int(1000 * bbox[3]),
    ]

