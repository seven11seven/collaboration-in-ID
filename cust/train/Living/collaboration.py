"""
1. Data packages for 4 clients {1: 3, 2: 2, 3: 2, 4: 1 (bad)} each with 128 drawings
2. Pretrained models on self-owned data
3. Data declaration
4. Make orders and create colations
5. Federated learning on colations
6. Records
"""
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


######################################
# Bad dataloader
######################################
class BadSplitLivingDataset(data.Dataset):
    """
    Given data_root and idxs,
    Generate class Dataset for net training.
    """
    def __init__(self, data_root, mask_size, idxs):
        """
        @param data_root: pickle dictory
        @param mask_size: int, seems not so important
        @param idxs: set/list of int
        """
        self.mask_size = mask_size
        self.floorplans = [os.path.join(data_root, pth_path) for pth_path in os.listdir(data_root)]
        self.idxs = idxs
        assert len(self.idxs) < len(self.floorplans)
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, index):
        floorplan_path = self.floorplans[self.idxs[index]]
        floorplan = LoadFloorplanTrain(floorplan_path, self.mask_size)
        # inputs: x, traget: y
        inputs = floorplan.get_composite_living(num_extra_channels=0)
        living_h, living_w  = floorplan.living_node["centroid"]
        target = t.zeros(2)
        target[0] = living_h + 10
        target[1] = living_w - 10
        return inputs, target


######################################
# Set data packages for users
######################################
def make_data_package(data_iid: dict, data_root):
    """ 
    @param: data_iid: {0: 3, 1: 2, 2: 2, 3: 1}
    @return: user_dp: {0:[0, 1, 2], 1: [3, 4], ...}
    @return: dps    : {0:{...}, 1:{...}, ..., 7:{...}}
    """
    user_dp = {}
    dp_id = 0
    for user_id, dp_num in data_iid.items():
        user_dp[user_id] = [dp_id+i for i in range(dp_num)]
        dp_id += dp_num
    dps = iid_idxs(data_root=data_root, num_users=dp_id)
    return user_dp, dps


#####################################
# Train model by federated learning and local updating
#####################################
def trainer(model, connect, dataloader, epoch, device, opt):
    """
    Train model and connect on dataloader
    @param model, connect: nn.Module
    @param dataloader: eval dataloader
    @param epoch: epochs on train data
    @param device: trained device
    @return model parameters, connect parameters, epoch_loss: trained model and losses on each epoch
    """
    # Set to train model
    model.to(device)
    connect.to(device)
    model.train()
    connect.train()
    epoch_loss = []
    # Set optimizer
    lr = opt.lr_base
    optimizer = torch.optim.Adam(
        list(model.parameters())+list(connect.parameters()),
        lr = lr,
        weight_decay = opt.weight_decay
    )
    criterion = nn.SmoothL1Loss()
    # Training on local machine
    for iter in range(epoch):
        batch_loss = []
        for batch_idx, (inputs, target) in tqdm(enumerate(dataloader)):
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            score_model = model(inputs)
            score_connect = connect(score_model)
            loss = criterion(score_connect, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            if (batch_idx+1) % 20 == 0:
                print("batch loss: {}".format(sum(batch_loss[-20:])/20))
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    return model, connect, epoch_loss


def fd_trainer(global_model, global_connect, dataloaders, global_epoch, local_epoch, device, opt, fo):
    """
    federated learning trainer 
    @param global_model, global_connect: init networks
    @param dataloaders: list of dataloader
    @param global_epoch: global epoch for fd
    @param local_epoch: local epoch for fd
    @param device, opt:
    @param fo: file handler
    @return model, connect
    """
    global_model.to(device)
    global_connect.to(device)
    global_model.train()
    global_connect.train()
    # Copy weights
    global_model_weights = global_model.state_dict()
    global_connect_weights = global_connect.state_dict()
    # Making logger
    fo.write("Global epoch, User, Avg loss\n")
    fo.flush()
    for epoch in range(global_epoch):
        local_model_weights, local_connect_weights = [], []
        for i, dataloader in enumerate(dataloaders):
            # Train on local devices
            local_model, local_connect, epoch_loss = trainer(
                model=copy.deepcopy(global_model), 
                connect=copy.deepcopy(global_connect), 
                dataloader=dataloader, 
                epoch=local_epoch, 
                device=device, 
                opt=opt)
            local_model_weights.append(copy.deepcopy(local_model.state_dict()))
            local_connect_weights.append(copy.deepcopy(local_connect.state_dict()))
            print("global epoch: {}, local user: {}, local loss: {}".format(epoch, i, epoch_loss[-1]))
            fo.write("{}, {}, {}\n".format(epoch, i, epoch_loss[-1]))
            fo.flush()
        # Update global weights
        global_model_weights = average_weights(local_model_weights)
        global_connect_weights = average_weights(local_connect_weights)
        global_model.load_state_dict(global_model_weights)
        global_connect.load_state_dict(global_connect_weights)
    # Close file
    fo.close()
    return global_model, global_connect


def evaluater(model, connect, dataloader, device):
    """
    Evaluate trained model and connect on dataloader
    @param model, connect: nn.Module
    @param dataloader: eval dataloader
    @return val_error: float
    """
    model.to(device)
    connect.to(device)
    model.eval()
    connect.eval()
    error_meter = meter.AverageValueMeter()

    for _, (inputs, target) in enumerate(dataloader):
        batch_size = inputs.shape[0]
        with torch.no_grad():
            score_model = model(inputs)
            score_connect = connect(score_model)
        inputs, target = inputs.to(device), target.to(device)
        output = score_connect.cpu().numpy().astype(int)
        target = target.cpu().numpy()
        distance_error = (output[:,0]-target[:,0])**2 + (output[:,1]-target[:,1])**2
        for i in range(batch_size): 
            error_meter.add(distance_error[i]**0.5 * utils.pixel2length)

    model.train()
    connect.train()
    val_error = round(error_meter.value()[0], 5)

    return val_error


#####################################
# federated learning class for a client
#####################################
class FLTrainer(object):
    """
    Trainer for the client
    1. train model on local data packages
    2. train model on local data package individually
    3. train model by federated learning individually
    4. train model by federated learning wholely
    """
    def __init__(self, data_root, device, user_dp, dps, user_id, opt):
        self.device = device
        self.mask_size = opt.mask_size
        self.data_root = data_root
        self.user_dp = user_dp
        self.dps = dps
        self.user_id = user_id
        self.self_dps = self.user_dp[self.user_id]
        self.opt = opt

    def create_dataloader(self, dps_id: list=None):
        """
        Given data idxs created train dataloader.
        @param: fd_dps_id: id list of data packages
        """
        idxs = {}
        for dp_id in dps_id:
            idxs += self.dps[dp_id]
        idxs = list(idxs)    
        return DataLoader(SplitLivingDataset(self.data_root, self.mask_size, idxs), batch_size=self.opt.batch_size, shuffle=True)
    
    def train_self_all(self, model, connect, epoch):
        """
        Train model on all self owned data packages
        @param model, connect: NN models
        @param epoch: train epochs
        """
        dataloader = self.create_dataloader(dps_id=self.self_dps)
        model, connect, epoch_loss = trainer(model=model, connect=connect, dataloader=dataloader, epoch=epoch, device=self.device, opt=self.opt)
        return model, connect, epoch_loss
    
    def train_self_ind(self, model, connect, epoch, dp_id: int):
        """
        Train model on data package dp_id
        @param dp_id: id of a data package 
        """
        dataloader = self.create_dataloader(dps_id=[dp_id])
        model, connect, epoch_loss = trainer(model=model, connect=connect, dataloader=dataloader, epoch=epoch, device=self.device, opt=self.opt)
        return model, connect, epoch_loss

    def train_fd(self, model, connect, global_epoch, local_epoch, fd_dps_id, fo):
        """
        Train model on self data packages and fd packages
        @param model, connect : init global model and connect
        @param dps_id: id list of data packages
        """
        # Add self owned dataloader
        selfloader = self.create_dataloader(dps_id=self.self_dps)
        dataloaders = [selfloader]
        # Add fd dataloaders
        for dp_id in fd_dps_id:
            data_loader = self.create_dataloader(dps_id=[dp_id])
            dataloaders.append(data_loader)
        # FD training
        global_model, global_connect = fd_trainer(
            global_model=model, 
            global_connect=connect, 
            dataloaders=dataloaders, 
            global_epoch=global_epoch, 
            local_epoch=local_epoch, 
            device=self.device, 
            opt=self.opt, 
            fo=fo)
        return global_model, global_connect


def main(opt, device, user_dp, dps: Dict, validloader, data_root):
    """
    1. initialize pre- and post- performance of data packages
    2. train on self-owned dataset
    3. train by federated learning
    ##############
    1. logger
    2. save model
    """    
    #### Init pre- and post- performance of data packages
    # Open logger
    fo = open("log/inital_val.txt", "w")
    fo.write("dp_id, pre_error, post_error\n")
    fo.flush()
    for dp_id, idxs in dps.items():
        # make dataloader
        idxs = list(idxs)
        if dp_id == 7:
            dataloader = DataLoader(BadSplitLivingDataset(data_root, opt.mask_size, idxs), batch_size=opt.batch_size, shuffle=True)
        else:
            dataloader = DataLoader(SplitLivingDataset(data_root, opt.mask_size, idxs), batch_size=opt.batch_size, shuffle=True)
        
        # initialize model and connect
        model_path, connect_path = None, None
        model, connect = load_model(model_path, connect_path)
        model.to(device)
        connect.to(device)
        model.train()
        connect.train()
        pre_error = evaluater(model=model, connect=connect, dataloader=dataloader, device=device)
        model, connect, epoch_loss = trainer(model=model, connect=connect, dataloader=dataloader, epoch=8, device=device, opt=opt)
        post_error = evaluater(model=model, connect=connect, dataloader=dataloader, device=device)
        print("dp_id: {}, pre_error: {}, post_error: {}".format(dp_id, pre_error, post_error))
        fo.write("{},{},{}\n".format(dp_id, pre_error, post_error))
        fo.flush()



if __name__ == "__main__":
    # Parameters
    opt = LivingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "../../pickle/train"
    data_iid = {0: 3, 1: 2, 2: 2, 3: 1}
    user_dp, dps = make_data_package(data_iid, data_root)

    # Get evaluation dataloader
    validloader = DataLoader(
        LivingDataset(data_root=opt.val_data_root, mask_size=opt.mask_size), 
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers)
    
    # Train and log
    main(opt=opt, device=device, user_dp=user_dp, dps=dps, data_root=data_root, validloader=validloader)