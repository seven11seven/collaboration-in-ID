"""
Wall larger than 3 pixels will be fixed to 3
Image size: 256 * 256
"""
from vec_fix import extract_wall, extract_ext_wall, extract_out_door
import numpy as np
from PIL import Image
from typing import List, Dict
import matplotlib.pyplot as plt 
import os


wall_width = 6


def hor_wall(ch2: np.ndarray) -> np.ndarray:
    """
    Given an image with walls whose width from 3 to 6 pixels.
    Walls are 0 in ch2, while others not.
    @return fixed_wall: an one-channel image with 3 pixel-width horizental wall.
    """
    is_wall = (ch2==0)
    full_wall = np.ones_like(is_wall)*255
    fixed_wall = np.ones_like(is_wall)*255
    for i in range(is_wall.shape[0]):
        for j in range(is_wall.shape[1]):
            if is_wall[i, j]:
                # Judge whether pixel i,j is in horizental wall
                # Judge evidence: 2 < sum(continueous[i(+-5), j] == 0) < 7
                # Upside:
                k, mem_1 = 0, 0
                while k<wall_width:
                    k += 1
                    if (i-k)<0:
                        break
                    if not is_wall[i-k, j]:
                        break
                    if is_wall[i-k, j]:
                        mem_1 += 1
                # Downside
                k, mem_2 = 0, 0
                while k<wall_width:
                    k += 1
                    if (i+k)>256:
                        break
                    if not is_wall[i+k, j]:
                        break
                    if is_wall[i+k, j]:
                        mem_2 += 1
                # Judge evidence
                if (1<mem_1+mem_2) & (mem_1+mem_2<wall_width):
                    full_wall[i, j] = 0
                    if mem_1==mem_2:
                        fixed_wall[i, j] = 0
                    elif mem_1 == mem_2+1:
                        fixed_wall[i, j] = 0
    return full_wall, fixed_wall


def ver_wall(ch2: np.ndarray) -> np.ndarray:
    """
    Given an image with walls whose width from 3 to 6 pixels.
    Walls are 0 in channel 2, while others not.
    @return fixed_wall: an one-channel image with 3 pixel-width vertical wall.
    """
    is_wall = (ch2==0)
    full_wall = np.ones_like(is_wall)*255
    fixed_wall = np.ones_like(is_wall)*255
    for i in range(is_wall.shape[0]):
        for j in range(is_wall.shape[1]):
            if is_wall[i, j]:
                # Judge whether pixel i,j is in horizental wall
                # Judge evidence: 2 < sum(continueous[i, j(+-)5] == 0) < 7
                # Left side:
                k, mem_1 = 0, 0
                while k<wall_width:
                    k += 1
                    if (j-k)<0:
                        break
                    if not is_wall[i, j-k]:
                        break
                    if is_wall[i, j-k]:
                        mem_1 += 1
                # Right side
                k, mem_2 = 0, 0
                while k<wall_width:
                    k += 1
                    if (j+k)>256:
                        break
                    if not is_wall[i, j+k]:
                        break
                    if is_wall[i, j+k]:
                        mem_2 += 1
                # Judge evidence
                if (1<mem_1+mem_2) & (mem_1+mem_2<wall_width):
                    full_wall[i, j] = 0
                    if mem_1==mem_2:
                        fixed_wall[i, j] = 0
                    elif mem_1 == mem_2+1:
                        fixed_wall[i, j] = 0
    return full_wall, fixed_wall


def reg_node(ch2: np.ndarray) -> Dict:
    full_ver_wall, fix_ver_wall = ver_wall(ch2=ch2)
    full_hor_wall, fix_hor_wall = hor_wall(ch2=ch2)
    all_wall = (ch2!=0)*255
    nodes = (all_wall==0)*255 - ((full_hor_wall==0) | (full_ver_wall==0))*255
    nodes = (nodes==0)*255
    is_nodes = (nodes==0)
    nodes_dic = {}
    for i in range(is_nodes.shape[0]):
        for j in range(is_nodes.shape[1]):
            if is_nodes[i, j]:
                # Get the center point of the node.
                x, y = 0, 0
                while is_nodes[i+y, j]:
                    y += 1
                while is_nodes[i, j+x]:
                    x += 1
                x_center = j+x//2
                y_center = i+y//2
                # Check the class of the node.
                has_left = (full_hor_wall[y_center, j-1]==0)
                has_right = (full_hor_wall[y_center, j+x]==0)
                has_up = (full_ver_wall[i-1, x_center]==0)
                has_down = (full_ver_wall[i+y, x_center]==0)
                if has_left and has_right and has_up and has_down:
                    nodes_dic[(x_center, y_center)] = 5
                elif has_left and has_right and has_down:
                    nodes_dic[(x_center, y_center)] = 2
                elif has_right and has_up and has_down:
                    nodes_dic[(x_center, y_center)] = 4
                elif has_left and has_up and has_down:
                    nodes_dic[(x_center, y_center)] = 6
                elif has_left and has_right and has_up:
                    nodes_dic[(x_center, y_center)] = 8
                elif has_right and has_down:
                    nodes_dic[(x_center, y_center)] = 1
                elif has_left and has_down:
                    nodes_dic[(x_center, y_center)] = 3
                elif has_right and has_up:
                    nodes_dic[(x_center, y_center)] = 7
                elif has_left and has_up:
                    nodes_dic[(x_center, y_center)] = 9
                # Delete this node.
                is_nodes[i:i+y, j:j+x] = False
    return nodes_dic


def reg_door(ch2: np.ndarray) -> List:
    is_door = (ch2==0)
    for i in range(is_door.shape[0]):
        for j in range(is_door.shape[1]):
            if is_door[i, j]:
                # Get the center point of the door.
                x, y = 0, 0
                while is_door[i+y, j]:
                    y += 1
                while is_door[i, j+x]:
                    x += 1
                x_center = j+x//2
                y_center = i+y//2
                door_node = [x_center, y_center]
                return door_node


def dic2img(ch2: np.ndarray, nodes_dict: Dict) -> np.ndarray:
    """
    Visualize nodes_dict on an image size like img
    """
    img = ch2
    for location, node in nodes_dict.items():
        j, i = location
        if node == 1:
            img[i:i+3, j] = 100
            img[i, j:j+3] = 100
        elif node == 2:
            img[i:i+3, j] = 100
            img[i, j-2:j+3] = 100
        elif node == 3:
            img[i:i+3, j] = 100
            img[i, j-2:j] = 100
        elif node == 4:
            img[i-2:i+3, j] = 100
            img[i, j:j+3] = 100
        elif node == 5:
            img[i-2:i+3, j] = 100
            img[i, j-2:j+3] = 100
        elif node == 6:
            img[i-2:i+3, j] = 100
            img[i, j-2:j] = 100
        elif node == 7:
            img[i-2:i, j] = 100
            img[i, j:j+3] = 100
        elif node == 8:
            img[i-2:i, j] = 100
            img[i, j-2:j+3] = 100
        elif node == 9:
            img[i-2:i, j] = 100
            img[i, j-2:j+1] = 100
        elif node == 0:
            img[i, j] = 100

    return img


def dic2list(nodes: Dict) -> List:
    """ nodes dic to sorted list """
    nodes_list = []
    for key, value in nodes.items():
        nodes_list.append([key[0], key[1], value])
    nodes_list = sorted(nodes_list, key = lambda x: 10*x[0]+0.1*x[1])
    return nodes_list


def train_pair():
    img_dir = "../data/dataset/floorplan_dataset"
    txt_dir = "../data/dataset/floorplan_txt"
    show_dir = "../data/dataset/floorplan_show"
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
    if not os.path.exists(show_dir):
        os.mkdir(show_dir)
    # Load image
    imgs = os.listdir(img_dir)
    for img_name in imgs:
        img = np.asarray(Image.open(os.path.join(img_dir, img_name)))
        # Extract input and output
        all_wall_ch2 = extract_wall(img)[:, :, 1]
        ext_wall_ch2 = extract_ext_wall(img)[:, :, 1]
        door_ch2 = extract_out_door(img)[:, :, 1]
        all_wall_nodes = reg_node(ch2=all_wall_ch2)
        ext_wall_nodes = reg_node(ch2=ext_wall_ch2)
        door_nodes = reg_door(ch2=door_ch2)
        all_wall_nodes[(door_nodes[0], door_nodes[1])] = 0
        ext_wall_nodes[(door_nodes[0], door_nodes[1])] = 0
        # Save image
        input_img = dic2img(ch2=ext_wall_ch2, nodes_dict=ext_wall_nodes)
        output_img = dic2img(ch2=all_wall_ch2, nodes_dict=all_wall_nodes)
        plt.imshow(input_img)
        plt.axis("off")
        plt.savefig(os.path.join(show_dir, img_name.split(".")[0]+"_in.png"))

        plt.imshow(output_img)
        plt.axis("off")
        plt.savefig(os.path.join(show_dir, img_name.split(".")[0]+"_out.png"))
        
        # Write to txt
        input_list = dic2list(ext_wall_nodes)
        output_list = dic2list(all_wall_nodes)
        with open(os.path.join(txt_dir, img_name.split(".")[0]+".txt"), "w") as wf:
            for line in input_list:
                wf.write("{},{},{};".format(line[0], line[1], line[2]))
            wf.write("\n")
            for line in output_list:
                wf.write("{},{},{};".format(line[0], line[1], line[2]))
            wf.write("\n")


if __name__ == "__main__":
    train_pair()