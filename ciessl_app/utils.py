import random
import os

import numpy as np
import torch, cv2
import json

from model.autoencoder import VoiceVAE, VoiceEncoder, VoiceConvAE


def show_flooding_map(map_data):
    cv2.imshow("flooding_map", map_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_segmented_map(map_data, n_room, room_centers, 
    origin, src, dst, boundary, output_path="segmented_map.png"):
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

    # crop out room only
    min_x = boundary["bl"][0]
    min_y = boundary["bl"][1]
    max_x = boundary["tr"][0]
    max_y = boundary["tr"][1]
    img = img[min_y:max_y, min_x:max_x]

    cv2.imwrite(output_path, img)


def load_encoder_model(cfg_path):
    json_file=open(cfg_path).read()
    data = json.loads(json_file)

    encoder_type = data["encoder_type"]

    # reconstruct model params file directory
    cwd = os.getcwd().split("/") + cfg_path.split("/")
    cwd[-1] = data["params_dir"]
    params_dir = '/'.join(cwd)

    encoder = None
    if encoder_type == "VoiceEncoder":
        encoder = VoiceEncoder(nn_structure=data["nn_structure"])
    elif encoder_type == "VoiceVAE":
        encoder = VoiceVAE(nn_structure=data["nn_structure"])
    elif encoder_type == "VoiceConvAE":
        encoder = VoiceConvAE(code_size=data["code_size"])
    else:
        print("[ERROR] utils.load_encoder_model(): do not support encoder type: {}".format(encoder_type))
        raise

    encoder.load(params_dir)

    return encoder


def label2rank(label, n_labels):
    new_y = []
    for i in range(1, n_labels + 1):
        new_y.append(1 if i == label else -1)
    new_y = np.asarray([new_y])

    return new_y


def init_training_set(data_loader, pipe, n_samples, seed=0, type="clf"):
    map_data = data_loader.load_map_info()
    init_X, init_y = None, None

    for voice in data_loader.voice_data_iterator(n_samples=n_samples, seed=seed):
        X, y = pipe.prepare_training_data(map_data, voice)

        if type == "rank":
            y = label2rank(y, map_data["n_room"])

        if init_X is None:
            init_X = X
            init_y = y
        else:
            init_X = np.append(init_X, X, axis=0)
            init_y = np.append(init_y, y, axis=0)

    return init_X, init_y


