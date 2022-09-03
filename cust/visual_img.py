"""
Visualize data set for analysis
"""
import numpy as np
from PIL import Image
from typing import List, Dict
import matplotlib.pyplot as plt 


def make_channel_one(img: np.ndarray) -> np.ndarray:
    """
    Transfer img into a four-channel image according to layer one
    exterior wall: 127 --> (0, 0, 0, 255)
    front door   : 255 --> (0, 255, 0, 255)
    other        : 0 --> (255, 255, 255, 255)
    """
    trans_img = np.zeros_like(img)
    ch_one = img[:, :, 0]
    # Visualize exterior wall
    trans_img[ch_one==127, 0] = 0
    trans_img[ch_one==127, 1] = 0
    trans_img[ch_one==127, 2] = 0
    trans_img[ch_one==127, 3] = 255
    # Visualize front door
    trans_img[ch_one==255, 0] = 0
    trans_img[ch_one==255, 1] = 255
    trans_img[ch_one==255, 2] = 0
    trans_img[ch_one==255, 3] = 255
    # Visualize other domain
    trans_img[ch_one==0, 0] = 255
    trans_img[ch_one==0, 1] = 255
    trans_img[ch_one==0, 2] = 255
    trans_img[ch_one==0, 3] = 255
    return trans_img


def make_channel_two(img: np.ndarray) -> np.ndarray:
    """
    Visualize 18 compoenents: 255/5 = 51, i = 0~17
    exterior wall: 14 --> (0, 0, 0, 255)
    front door   : 15 --> (0, 255, 0, 255)
    other        : 0 --> (255, 255, 255, 255)
    """
    trans_img = np.zeros_like(img)
    ch_two = img[:, :, 1]
    color_map = {
        0: [254,  67, 101], 9:  [220,  87,  18],
        1: [252, 157, 154], 10: [178, 200, 187],
        2: [249, 205, 173], 11: [117, 121,  74],
        3: [200, 200, 169], 12: [150, 150, 150],
        4: [131, 175, 155], 13: [255, 255, 255],
        5: [182, 194, 154], 14: [  0,   0,   0],
        6: [138, 151, 123], 15: [  0, 255,   0],
        7: [244, 208,   0], 16: [200, 200, 200],
        8: [229, 131,   8], 17: [ 64, 116,  52],
    }
    for i in range(18):
        trans_img[ch_two==i, 0] = color_map[i][0]
        trans_img[ch_two==i, 1] = color_map[i][1]
        trans_img[ch_two==i, 2] = color_map[i][2]
        trans_img[ch_two==i, 3] = 255
    return trans_img


def make_channel_three(img: np.ndarray) -> np.ndarray:
    """
    Channel three is the centroid of different rooms
    0, 100~110
    """
    trans_img = np.zeros_like(img)
    ch_three = img[:, :, 2]
    color_map = {
          0: [254,  67, 101], 100: [220,  87,  18],
        101: [252, 157, 154], 102: [178, 200, 187],
        103: [249, 205, 173], 104: [117, 121,  74],
        105: [200, 200, 169], 106: [150, 150, 150],
        107: [131, 175, 155], 108: [255, 255, 255],
        109: [182, 194, 154], 110: [  0,   0,   0],
    }
    for i in range(100, 111, 1):
        trans_img[ch_three==i, 0] = color_map[i][0]
        trans_img[ch_three==i, 1] = color_map[i][1]
        trans_img[ch_three==i, 2] = color_map[i][2]
        trans_img[ch_three==i, 3] = 255
    return trans_img


def make_channel_four(img: np.ndarray) -> np.ndarray:
    """
    Channel four is the of 
    0 --> ()
    225 --> ()
    """
    trans_img = np.zeros_like(img)
    ch_four = img[:, :, 3]
    color_map = {
          0: [254,  67, 101], 
        255: [220,  87,  18]
    }
    for i in [0, 255]:
        trans_img[ch_four==i, 0] = color_map[i][0]
        trans_img[ch_four==i, 1] = color_map[i][1]
        trans_img[ch_four==i, 2] = color_map[i][2]
        trans_img[ch_four==i, 3] = 255
    return trans_img


def visual_output(img: np.ndarray) -> np.ndarray:
    """
    Visualize output image:
    Channel one: exterior wall and door
    Channel two: 
    """
    


def static_img(img: np.ndarray) -> List:
    """
    Output layout:
    layer one:    out walls and front door
    layer two:    0, 127  ->  inner walls and other domain
    layer three: 100~110  ->  center of different rooms 
    layer four:   0, 255  ->  indoor area and outdoor area
    """
    img_dict = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r, g, b, d = img[i, j, :]
            if "{}-{}-{}-{}".format(r, g, b, d) in img_dict:
                img_dict["{}-{}-{}-{}".format(r, g, b, d)] += 1
            else:
                img_dict["{}-{}-{}-{}".format(r, g, b, d)] = 0    
    return [key.split("-") for key in img_dict.keys()]


if __name__ == "__main__":
    # img = np.asarray(Image.open("../data/sync-layout/0in1367.png"))
    # img = np.asarray(Image.open("../data/dataset/floorplan_dataset/1015.png"))
    img = np.asarray(Image.open("../data/synth_input/0in1015.png"))

    # print(static_img(img=img))
    img = make_channel_four(img)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    