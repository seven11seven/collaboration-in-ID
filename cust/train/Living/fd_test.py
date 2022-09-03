from config_living import LivingConfig
from data.dataset_living import *
from torch.utils.data import DataLoader
from torchnet import meter
from torch import nn
from tqdm import tqdm
import models
import torch
import copy


opt = LivingConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def iid_idxs(data_root, num_users):
    """
    Split data into num_users idxs
    @param data_root: pickle data folder
    @param num_users: number of data owner
    @return dict_users: { user_id: {pickle_id, pickle_id, ...} }
    """
    num_items = int(len(os.listdir(data_root))/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(os.listdir(data_root)))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def average_weights(w):
    """
    Given parameters lists of different user models w,
    Return average parameters.
    @param w: List: [w[0], w[1], w[2], ...]
              w[i]: Dict: {}
    @return w_avg: sum(w)/len(w)
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class SplitLivingDataset(data.Dataset):
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
        target[0] = living_h
        target[1] = living_w
        return inputs, target


def load_model(model_path=None, connect_path=None):
    """
    Init/load model and connect
    @param model_path, connect_path: model and connect model path 
    @return model, connect 
    """
    # initialize model
    model = models.model(
        module_name = opt.module_name,
        model_name = opt.model_name,
        input_channel = 3,
        output_channel = 2,
        pretrained = False
    )
    input_channel = 512
    connect = models.connect(
        module_name = opt.module_name,
        model_name = opt.model_name,
        input_channel = input_channel,
        output_channel = 2,
        reshape = True
    )
    # load model
    if model_path is not None:
        model.load_model(path=model_path)
    if connect_path is not None:
        connect.load_model(path=connect_path)
    return model, connect


def evaluate(model, connect, dataloader, device):
    """
    Evaluate trained model and connect on dataloader
    @param model, connect: nn.Module
    @param dataloader: eval dataloader
    @return val_error: float
    """
    model.to(device)
    connet.to(device)
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


def train(model, connect, dataloader, epoch, device):
    """
    Train model and connect on dataloader
    @param model, connet: nn.Module
    @param dataloader: eval dataloader
    @param epoch: epochs on train data
    @param device: trained device
    @return model parameters, connect parameters, epoch_loss: trained model and epoch loss
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

    return model, connect, epoch_loss[-1]


class LocalUpdate(object):
    """
    A local training process to get weights on a single data owner.
    deepcopy of global model --> local model
    dataset + idxs --> trainloader, testloader to update local model    
    """
    def __init__(self, data_root, mask_size, idxs):
        """
        @param data_root: dictory of pickle files
        @param mask_size: not so important 
        @param idxs:      idxs locating pickle files
        """
        # self.criterion = nn.SmoothL1Loss()
        self.trainloader, self.validloader = self.train_val(data_root, mask_size, list(idxs))
        self.device = device

    def train_val(self, data_root, mask_size, idxs):
        """
        @param data_root: dictoty containing all pickle files
        @param mask_size: not so important
        @param idxs:      data idxs of data owner
        @return trainloader, validloader for data owner
        """
        # Split indexed for train, val, and test
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_val = idxs[int(0.9*len(idxs)):]
        # Wrrap dataset to DataLoader
        trainloader = DataLoader(SplitLivingDataset(data_root, mask_size, idxs_train), batch_size=opt.batch_size, shuffle=True)
        validloader = DataLoader(SplitLivingDataset(data_root, mask_size, idxs_val), batch_size=opt.batch_size, shuffle=False)
        return trainloader, validloader
    
    def update_weights(self, model, connect, local_epoch):
        """
        @param model, connect: NN module copied from global model
        @param global_round  : not so important now
        @return model parameters, connect parameters, average epoch loss 
        """
        model, connect, train_loss = train(model=model, connect=connect, dataloader=self.trainloader, epoch=local_epoch, device=self.device)
        val_error = evaluate(model=model, connect=connect, dataloader=self.validloader, device=self.device)
        return model.state_dict(), connect.state_dict(), train_loss, val_error


def fd_learn():
    # Set global model
    model_path, connect_path = None, None
    global_model, global_connect = load_model(model_path, connect_path)
    global_model.to(device)
    global_connect.to(device)
    global_model.train()
    global_connect.train()
    # Copy weights
    global_model_weights = global_model.state_dict()
    global_connect_weights = global_connect.state_dict()
    # Create evaluation dataloader
    validloader = DataLoader(
        LivingDataset(data_root=opt.val_data_root, mask_size=opt.mask_size), 
        batch_size=opt.batch_size, 
        shuffle=False,
        num_workers=opt.num_workers
        )
    # Prepare data
    # Make dataset loader
    num_users = 3
    data_root = "../../pickle/train"
    dict_users = iid_idxs(data_root=data_root, num_users=num_users)
    mask_size = opt.mask_size
    global_epoch = 20
    local_epoch = 5
    # Make logger
    fo = open("fd_living.csv", "w")
    fo.write("Global epoch, User, Avg loss, Val loss\n")
    fo.flush()
    for epoch in range(global_epoch):
        local_model_weights, local_connect_weights = [], []
        # All three users will be participated
        # for each data owners, get an updated weights and calculate their average
        for idx in range(num_users):
            local_model = LocalUpdate(data_root=data_root, mask_size=mask_size, idxs=dict_users[idx])
            model_weights, connect_weights, loss, val_error = local_model.update_weights(
                model=copy.deepcopy(global_model), 
                connect=copy.deepcopy(global_connect), 
                local_epoch=local_epoch
            )
            local_model_weights.append(copy.deepcopy(model_weights))
            local_connect_weights.append(copy.deepcopy(connect_weights))
            print("global epoch: {}, local user: {}, local loss: {}, local error{}".format(epoch, idx, loss, val_error))
            fo.write("{}, {}, {}, {}\n".format(epoch, idx, loss, val_error))
            fo.flush()
        # update global weights
        global_model_weights = average_weights(local_model_weights)
        global_connect_weights = average_weights(local_connect_weights)
        global_model.load_state_dict(global_model_weights)
        global_connect.load_state_dict(global_connect_weights)
        # global evaluation
        val_error = evaluate(model=global_model, connect=global_connect, dataloader=validloader, device=device)
        print("global epoch: {}, value error: {}".format(epoch, val_error))
        fo.write("{}, {}, {}, {}\n".format(epoch, "global", "-", val_error))
        fo.flush()
        if val_error < 1.6:
            global_model.save_model(2000)
            global_connect.save_model(2000)
    # Save final model
    fo.close()
    global_model.save_model(1000)
    global_connect.save_model(1000)


if __name__ == "__main__":
    fd_learn()
