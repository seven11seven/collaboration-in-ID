from typing import Tuple


room_label = [
    (0,'LivingRoom'), (1,'MasterRoom'), (2,'Kitchen'),
    (3,'Bathroom'), (4,'DiningRoom'), (5,'ChildRoom'),
    (6,'StudyRoom'), (7,'SecondRoom'), (8,'GuestRoom'),
    (9,'Balcony'), (10,'Entrance'), (11,'Storage'),
    (12,'Wall-in'), (13,'External'), (14,'ExteriorWall'),
    (15,'FrontDoor'), (16,'InteriorWall'), (17,'InteriorDoor')
    ]


category = [category for category in room_label if category[1] not in set(["External", "ExteriorWall", "FrontDoor", "InteriorWall", "InteriorDoor"])]
num_category = len(category)
pixel2length = 18/256


def label2name(label: int = 0) -> str:
    """
    Given label index to relevant name
    """
    if label<0 or label>17:
        raise Exception("Invalid label: ", label)
    else:
        return room_label[label][1]


def laebl2index(label: int = 0) -> int:
    """
    Return a label if it's legal
    """
    if label<0 or label>17:
        raise Exception("Invalid label: ", label)
    else:
        return label


def compute_centroid(mask) -> Tuple[int, int]:
    """
    Given a masked region, return its center coordinations
    """
    sum_h, sum_w, count = 0, 0, 0
    shape_array = mask.shape
    for h in range(shape_array[0]):
        for w in range(shape_array[1]):
            if mask[h, w] != 0:
                sum_h += h
                sum_w += w
                count += 1
    return sum_h//count, sum_w//count


def log(file, msg="", is_print=True):
    """
    Log msg to file
    """
    if is_print:
        print(msg)
    file.write(msg + "\n")
    file.flush()
