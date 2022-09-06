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


def compute_centroid(mask):
    sum_h = 0
    sum_w = 0
    count = 0
    shape_array = mask.shape
    for h in range(shape_array[0]):  
        for w in range(shape_array[1]):
            if mask[h, w] != 0:
                sum_h += h
                sum_w += w
                count += 1
    return (sum_h//count, sum_w//count)


def show_centroid(path):
    with Image.open(path) as temp:
        image_array = np.asarray(temp, dtype=np.uint8)
    boundary_mask = image_array[:,:,0]
    category_mask = image_array[:,:,1]
    index_mask = image_array[:,:,2]
    inside_mask = image_array[:,:,3]
    shape_array = image_array.shape
    index_category = []
    room_node = []

    interiorWall_mask = np.zeros(category_mask.shape, dtype=np.uint8)
    interiorWall_mask[category_mask == 16] = 1        
    interiordoor_mask = np.zeros(category_mask.shape, dtype=np.uint8)
    interiordoor_mask[category_mask == 17] = 1

    for h in range(shape_array[0]):  
        for w in range(shape_array[1]):
            index = index_mask[h, w]
            category = category_mask[h, w]
            if index > 0 and category <= 12:
                if len(index_category):
                    flag = True
                    for i in index_category:
                        if i[0] == index:
                            flag = False
                    if flag:
                        index_category.append((index, category))
                else:
                    index_category.append((index, category))

    for (index, category) in index_category:
        node = {}
        node['category'] = int(category)
        mask = np.zeros(index_mask.shape, dtype=np.uint8)
        mask[index_mask == index] = 1
        node['centroid'] = compute_centroid(mask)
        room_node.append(node)
    
    return room_node



def add_center(path):
    img = np.asarray(Image.open(path))
    img = make_channel_two(img)

    nodes = show_centroid(path=path)
    for node in nodes:
        x, y = node["centroid"]
        img[x-1:x+2, y-1:y+2, :] = [0, 255, 255, 255]

    plt.imshow(img)
    plt.axis("off")
    plt.show()



if __name__ == "__main__":
    # img = np.asarray(Image.open("../data/sync-layout/0in1367.png"))
    # img = np.asarray(Image.open("../data/dataset/floorplan_dataset/1015.png"))
    # img = np.asarray(Image.open("./plot/1000.png"))
    # img = make_channel_two(img)
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()

    add_center(path="./plot/1000.png")
