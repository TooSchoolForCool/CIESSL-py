import random

import numpy as np
import cv2


def show_flooding_map(map_data):
    cv2.imshow("flooding_map", map_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_segmented_map(map_data, n_room, room_centers, 
    origin, src, dst, output_path="segmented_map.png"):
    """
    Save segmented map image to local

    Args:
        output_path (string): output path of the image
    """
    img = np.zeros((map_data.shape[0], map_data.shape[1], 3), np.uint8)

    # paint different room with random color
    for i in range(1, n_room + 1):
        blue = random.randint(0, 256)
        green = random.randint(0, 256)
        red = random.randint(0, 256)

        # traverse each 
        for x in range(0, map_data.shape[0]):
            for y in range(0, map_data.shape[1]):
                if map_data[x, y] == i:
                    img[x, y] = (blue, green, red)

    # paint estimated room center
    for c in room_centers:
        cv2.circle(img, c, 2, (255, 0, 0), -1)

    # paint origin of the map
    cv2.drawMarker(img, origin, (255, 255, 255), markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=4)

    # paint sound source locations
    for s in src:
        cv2.circle(img, s, 2, (0, 0, 255), -1)

    # paint microphone location
    for d in dst:
        cv2.circle(img, d, 2, (255, 255, 255), -1)

    cv2.imwrite(output_path, img)