from typing import Dict
from config_living import LivingConfig
from data.dataset_living import *
from torch.utils.data import DataLoader
from torchnet import meter
from torch import nn
from tqdm import tqdm
import models
import torch
import copy

from fd_test import iid_idxs, average_weights, SplitLivingDataset, load_model


if __name__ == "__main__":
    opt = LivingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "../../pickle/train"
    
    validloader = DataLoader(
        LivingDataset(data_root=opt.val_data_root, mask_size=opt.mask_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers)
    
    model_path, connect_path = None, None
    model, connect = load_model(model_path, connect_path)
    model.to(device)
    connect.to(device)
    model.eval()
    connect.eval()

    for _, (inputs, target) in enumerate(validloader):
        batch_size = inputs.shape[0]
        inputs, target = inputs.to(device), target.to(device)
        with torch.no_grad():
            score_model = model(inputs)
            score_connect = connect(score_model)
        output = score_connect.cpu().numpy().astype(int)
        target = target.cpu().numpy()
        print(output)
        print(target)