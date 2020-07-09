# -*- coding:utf-8 -*-

import torch,config,os
import torch.optim as optim
import torch.nn as nn
import core.simple_model as model
import core.data_handler as dataHandler
from torch.utils.data import DataLoader

os.environ["TORCH_HOME"] = config.ENV
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(6878237888968)

def check_gpu(f):
    def wrapper(*args,**kargs):
         import pynvml,time
         pynvml.nvmlInit()

         while(True):
             handle = pynvml.nvmlDeviceGetHandleByIndex(0)
             mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
             mem_free = mem_info.free / (1024 ** 3)
             time_str = time.asctime(time.localtime(time.time()))
             if mem_free >= 4:
                 print("[",time_str,"]","It is time to train!")
                 return f(*args,**kargs)
                 # notify_me()
                 #  break
             else:
                 print("[",time_str,"]","Memory only remains ",mem_free," GB")
             time.sleep(6)
    return wrapper

# @check_gpu
def train():
    train_data_set = dataHandler.CUBDataset(data_root=config.DATA_ROOT,training=True, resize=config.RESIZE_SIZE, crop=config.CROP_SIZE)
    train_data_loader = DataLoader(train_data_set,batch_size=config.BATCH_SIZE, shuffle=True,num_workers=config.WORKER_NUM,drop_last=False)
    valid_data_set = dataHandler.CUBDataset(data_root=config.DATA_ROOT, training=False,resize=config.RESIZE_SIZE, crop=config.CROP_SIZE)
    valid_data_loader = DataLoader(valid_data_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.WORKER_NUM,drop_last=False)

    net = model.IERNet(num_classes=200).to(device)

    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(),lr=config.LR,momentum=0.9, weight_decay=config.WD)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.96)

    # Training
    print("-" * 60)
    print("Start Training:")
    train_acc_list = []
    valid_acc_list = []
    for epoch in range(config.EPOCH):

        net.train()
        train_acc_num = 0
        train_total_num = 0
        valid_acc_num = 0
        valid_total_num = 0
        train_cls_loss = 0
        train_reg_loss = 0

        for i,data in enumerate(train_data_loader):

            optimizer.zero_grad()

            img, target, entropy = data
            img = img.to(device)
            target = target.to(device)
            entropy = entropy.to(device)

            fusion_cls_logits,feature_raw_cls_logits,attention1_cls_logits,extracted_feature_entropy,attention1_maps_entropy = net(img, True)

            fusion_cls_loss = criterion_cls(fusion_cls_logits, target)
            feature_raw_cls_loss = criterion_cls(feature_raw_cls_logits, target)
            attention1_cls_loss = criterion_cls(attention1_cls_logits, target)

            feature_entropy_loss = criterion_reg(extracted_feature_entropy, entropy.float())
            attention1_entropy_reg_loss = criterion_reg(attention1_maps_entropy, (1 - config.ENTROPY_SCALE) * entropy.float())

            cls_loss =  fusion_cls_loss + feature_raw_cls_loss + attention1_cls_loss
            entropy_reg_loss = feature_entropy_loss + attention1_entropy_reg_loss

            loss = cls_loss  + 0.001 *  entropy_reg_loss
            loss.backward()

            optimizer.step()
            train_cls_loss += cls_loss.item()
            train_reg_loss += entropy_reg_loss.item()

            # statistic
            prob = torch.log_softmax(fusion_cls_logits,dim=1)
            train_acc_num += (torch.sum(torch.argmax(prob,dim=1) == target)).item()
            train_total_num += target.size()[0]

            if i % 10 == 1:
                print("Epoch :",epoch, ",Step:",i, ",Total Loss: ",
                      (train_cls_loss + train_reg_loss) / 10,
                      ",Classification Loss:",train_cls_loss / 10,",Regression Loss:",train_reg_loss / 10)
                train_cls_loss = 0
                train_reg_loss = 0

        # schedule
        lr_scheduler.step()

        # validation
        with torch.no_grad():
            net.eval()
            for i, data in enumerate(valid_data_loader):
                img, target, _ = data
                img = img.to(device)
                target = target.to(device)

                fusion_cls_logits, _, _, _,_  = net(img, False)

                prob = torch.log_softmax(fusion_cls_logits, dim=1)
                valid_acc_num += (torch.sum(torch.argmax(prob, dim=1) == target)).item()
                valid_total_num += target.size()[0]

        train_acc_list.append(train_acc_num / train_total_num)
        valid_acc_list.append(valid_acc_num / valid_total_num)
        print("Epoch:",epoch,"Training top1 acc:",train_acc_num/train_total_num)
        print("Epoch:",epoch,"Validation top1 acc:",valid_acc_num/valid_total_num)
        print(train_acc_list)
        print(valid_acc_list)
        torch.save(net, os.path.join(config.MODEL_SAVE_PATH,"ier_net_"+ str(epoch) + ".pth"))

if __name__ == "__main__":

    train()
