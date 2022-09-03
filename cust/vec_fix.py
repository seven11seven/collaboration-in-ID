"""
Transfer layout image into graph name
"""

from visual_img import *
from typing import Dict, List
import numpy as np
import os


# Wall: 12, 14, 15, 16, 17
room_label = [(0,'LivingRoom'),
              (1,'MasterRoom'),
              (2,'Kitchen'),
              (3,'Bathroom'),
              (4,'DiningRoom'),
              (5,'ChildRoom'),
              (6,'StudyRoom'),
              (7,'SecondRoom'),
              (8,'GuestRoom'),
              (9,'Balcony'),
              (10,'Entrance'),
              (11,'Storage'),
              (12,'Wall-in'),
              (13,'External'),
              (14,'ExteriorWall'),
              (15,'FrontDoor'),
              (16,'InteriorWall'),
              (17,'InteriorDoor')]


def extract_wall(img: np.ndarray) -> np.ndarray:
    """
    img_size: 256 ~ 256
    Visualize 18 compoenents: 255/5 = 51, i = 0 ~ 17
    exterior wall: 14 --> (0, 0, 0, 255)
    front door   : 15 --> (0, 255, 0, 255)
    other        : 0  --> (255, 255, 255, 255)
    """
    trans_img = np.zeros_like(img)
    ch_two = img[:, :, 1]
    color_map = {
        0: [254,  67, 101], 9:  [220,  87,  18],
        1: [252, 157, 154], 10: [178, 200, 187],
        2: [249, 205, 173], 11: [117, 121,  74],
        3: [200, 200, 169], 12: [  0,   0,   0],
        4: [131, 175, 155], 13: [255, 255, 255],
        5: [182, 194, 154], 14: [  0,   0,   0],
        6: [138, 151, 123], 15: [  0,   0,   0],
        7: [244, 208,   1], 16: [  0,   0,   0],
        8: [229, 131,   8], 17: [  0,   0,   0],
    }
    for i in range(18):
        trans_img[ch_two==i, 0] = color_map[i][0]
        trans_img[ch_two==i, 1] = color_map[i][1]
        trans_img[ch_two==i, 2] = color_map[i][2]
        trans_img[ch_two==i, 3] = 255
    return trans_img


def extract_ext_wall(img: np.ndarray) -> np.ndarray:
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
    trans_img[ch_one==255, 1] = 0
    trans_img[ch_one==255, 2] = 0
    trans_img[ch_one==255, 3] = 255
    # Visualize other domain
    trans_img[ch_one==0, 0] = 255
    trans_img[ch_one==0, 1] = 255
    trans_img[ch_one==0, 2] = 255
    trans_img[ch_one==0, 3] = 255
    return trans_img


def extract_out_door(img: np.ndarray) -> np.ndarray:
    """
    Transfer img into a four-channel image according to layer one
    exterior wall: 127 --> (0, 0, 0, 255)
    front door   : 255 --> (0, 255, 0, 255)
    other        : 0 --> (255, 255, 255, 255)
    """
    trans_img = np.zeros_like(img)
    ch_one = img[:, :, 0]
    # Visualize exterior wall
    trans_img[ch_one==127, 0] = 255
    trans_img[ch_one==127, 1] = 255
    trans_img[ch_one==127, 2] = 255
    trans_img[ch_one==127, 3] = 255
    # Visualize front door
    trans_img[ch_one==255, 0] = 0
    trans_img[ch_one==255, 1] = 0
    trans_img[ch_one==255, 2] = 0
    trans_img[ch_one==255, 3] = 255
    # Visualize other domain
    trans_img[ch_one==0, 0] = 255
    trans_img[ch_one==0, 1] = 255
    trans_img[ch_one==0, 2] = 255
    trans_img[ch_one==0, 3] = 255
    return trans_img


def get_node(img: np.ndarray) -> np.ndarray:
    """
    Return the center points of nodes
    @param  img  : wall's second channel is 0, while others not
    @return nodes: [[x_0, y_0], [x_1, y_1], ...]
    """
    ch2 = img[:, :, 1]

    # Node pattern:
    # 10001 11111 | 11111 11111 11111 11111 10001 10001 10001 10001 10001 11111 10001
    # 10001 00000 | 10000 00001 10000 00001 10000 00001 10000 00001 00000 00000 00000
    # 10001 00000 | 10000 00001 10000 00001 10000 00001 10000 00001 00000 00000 00000
    # 10001 00000 | 10000 00001 10000 00001 10000 00001 10000 00001 00000 00000 00000
    # 10001 11111 | 11111 11111 10001 10001 11111 11111 10001 10001 11111 10001 10001
    # Vertical and horizental pattern
    conv_ver = np.asarray([
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,1,1,1,0]
        ])
    conv_hor = np.asarray([
        [0,0,0,0,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,0,0,0,0]
        ])

    # Recognize vertical case
    def rec_ver(node: np.ndarray):
        mult = conv_ver * node
        if np.sum(mult)==0:
            if (node[0,0]>0) and (node[1,0]>0) and (node[2,0]>0) and (node[3,0]>0) and (node[4,0]>0):
                if (node[0,4]>0) and (node[1,4]>0) and (node[2,4]>0) and (node[3,4]>0) and (node[4,4]>0):
                    return True
    
    # Recognize horizental case
    def rec_hor(node: np.ndarray):
        mult = conv_hor * node
        if np.sum(mult)==0:
            if (node[0,0]>0) and (node[0,1]>0) and (node[0,2]>0) and (node[0,3]>0) and (node[0,4]>0):
                if (node[4,0]>0) and (node[4,1]>0) and (node[4,2]>0) and (node[4,3]>0) and (node[4,4]>0):
                    return True

    mask = np.ones((3, 3))
    centers = []
    wall = (ch2==0)
    for i in range(1, ch2.shape[0]-1):
        for j in range(1, ch2.shape[1]-1):
            if np.sum(ch2[i-1:i+2, j-1:j+2]*mask)==0:
                node = ch2[i-2:i+3, j-2:j+3]
                is_ver = rec_ver(node)
                is_hor = rec_hor(node)
                if not (is_ver or is_hor):
                    centers.append([i, j])
    return centers    


def show_node(img: np.ndarray, nodes: List) -> np.ndarray:
    """
    Show nodes on image img.
    @param img: an image as the bottom layer
    @param nodes: a list containing coordinates of nodes
    """
    for node in nodes:
        img[node[0], node[1], 0] = 64
        img[node[0], node[1], 1] = 116
        img[node[0], node[1], 2] = 52
    return img


def node_classify(img: np.ndarray, nodes: List) -> Dict:
    """
    classify the node pixels in nodes list.
    @param img: give reference of image size
    @param nodes: a list containing coordinates of nodes
    @return nodes_dic: {(x, y): node_type}
    """
    img = img*0
    img[:, :, 3] = 255
    for node in nodes:
        img[node[0], node[1], 0] = 100
        img[node[0], node[1], 1] = 100
        img[node[0], node[1], 2] = 100
    
    bounding = (img[:, :, 0]==100)
    conv = np.asarray([
        [  0,   0,  1,    0,    0],
        [  0,   0,  1,    0,    0],
        [0.1, 0.1,  1, 0.01, 0.01],
        [  0,   0, 10,    0,    0],
        [  0,   0, 10,    0,    0]
        ])
    nodes_dic = {}
    for i in range(bounding.shape[0]):
        for j in range(bounding.shape[1]):
            if bounding[i][j]:
                box = img[i-2:i+3, j-2:j+3, 0]
                pattern = np.sum(conv*box)
                if pattern == 2322:
                    nodes_dic[(i, j)] = 4
                elif pattern == 2122:
                    nodes_dic[(i, j)] = 1
                elif pattern == 2302:
                    nodes_dic[(i, j)] = 3
                elif pattern == 2320:
                    nodes_dic[(i, j)] = 5
                elif pattern == 322:
                    nodes_dic[(i, j)] = 7
                elif pattern == 2102:
                    nodes_dic[(i, j)] = 0
                elif pattern == 2120:
                    nodes_dic[(i, j)] = 2
                elif pattern == 302:
                    nodes_dic[(i, j)] = 6
                elif pattern == 320:
                    nodes_dic[(i, j)] = 8
    return nodes_dic


def dic2img(img: np.ndarray, nodes_dict: Dict) -> np.ndarray:
    """
    Visualize nodes_dict on an image size like img
    """
    img = img*0
    img[:, :, 3] = 255
    for location, node in nodes_dict.items():
        i, j = location
        if node == 0:
            img[i:i+3, j, :] = 100
            img[i, j:j+3, :] = 100
        elif node == 1:
            img[i:i+3, j, :] = 100
            img[i, j-2:j+3, :] = 100
        elif node == 2:
            img[i:i+3, j, :] = 100
            img[i, j-2:j, :] = 100
        elif node == 3:
            img[i-2:i+3, j, :] = 100
            img[i, j:j+3, :] = 100
        elif node == 4:
            img[i-2:i+3, j, :] = 100
            img[i, j-2:j+3, :] = 100
        elif node == 5:
            img[i:i+3, j, :] = 100
            img[i, j-2:j, :] = 100
        elif node == 6:
            img[i-2:i, j, :] = 100
            img[i, j:j+3, :] = 100
        elif node == 7:
            img[i-2:i, j, :] = 100
            img[i, j-2:j+3, :] = 100
        elif node == 8:
            img[i-2:i, j, :] = 100
            img[i, j-2:j+1, :] = 100
        elif node == 9:
            img[i, j, :] = 100

    return img


def train_pair(img: np.ndarray):
    """
    @param img: 
    """
    # Input: exterior wall nodes
    ext_img = extract_ext_wall(img)
    nodes = get_node(img=ext_img)
    input_nodes_dic = node_classify(img, nodes)
    # Ouput: all wall nodes
    all_img = extract_wall(img)
    nodes = get_node(img=all_img)
    output_nodes_dic = node_classify(img, nodes)
    # Door: only one door in the image
    door_img = extract_out_door(img)
    nodes = get_node(img=door_img)
    loc = np.mean(np.asarray(nodes), axis=0)
    loc = loc.tolist()
    loc = (int(loc[0]), int(loc[1]))

    input_nodes_dic[loc] = 9
    output_nodes_dic[loc] = 9

    input_list = dic2list(input_nodes_dic)
    output_list = dic2list(output_nodes_dic)

    return input_list, output_list


def dic2list(nodes: Dict) -> List:
    """ nodes dic to sorted list """
    nodes_list = []
    for key, value in nodes.items():
        nodes_list.append([key[0], key[1], value])
    nodes_list = sorted(nodes_list, key = lambda x: 10*x[0]+0.1*x[1])
    return nodes_list


def main():
    img_dir = "../data/dataset/floorplan_dataset"
    txt_dir = "../data/dataset/floorplan_txt"
    imgs = os.listdir(img_dir)
    for img_name in imgs:
        img = np.asarray(Image.open(os.path.join(img_dir, img_name)))
        inputs, outputs = train_pair(img)
        with open(os.path.join(txt_dir, img_name.split(".")[0]+".txt"), "w") as wf:
            for line in inputs:
                wf.write("{},{},{};".format(line[0], line[1], line[2]))
            wf.write("\n")
            for line in outputs:
                wf.write("{},{},{};".format(line[0], line[1], line[2]))
            wf.write("\n")


if __name__ == "__main__":
    img = np.asarray(Image.open(os.path.join("../data/dataset/floorplan_dataset", "10645.png")))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
