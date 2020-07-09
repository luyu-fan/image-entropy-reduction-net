import os,PIL.Image as Image, numpy as np, math
import torchvision.transforms as transforms
import random
import torch

from torch.utils.data import Dataset,DataLoader

class CUBDataset(Dataset):

    def __init__(self,data_root,training=True,resize = 248,crop = 224):

        self.root = data_root
        self.is_train = training
        self.resize = resize
        self.crop = crop

        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))

        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        if self.is_train:
            self.train_img_path = [os.path.join(self.root, 'images', train_file) for train_file in
                                   train_file_list]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i]
        if not self.is_train:
            self.test_img_path = [os.path.join(self.root, 'images', test_file) for test_file in
                                  test_file_list]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i]

        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((self.resize,self.resize),Image.BILINEAR),
                transforms.CenterCrop(self.crop),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

    def __getitem__(self,index):

        entropy_r = entropy_g = entropy_b = 0

        if self.is_train:
            img = Image.open(self.train_img_path[index])
            img = img.resize((self.resize,self.resize),Image.BILINEAR)
            crop_h = (int)(random.uniform(0.1,0.95) * (self.resize - self.crop))
            crop_w = (int)(random.uniform(0.1,0.95) * (self.resize - self.crop))
            img = img.crop((crop_w, crop_h, crop_w + self.crop, crop_h + self.crop))
            prob = random.uniform(0,1)
            if prob >= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            target = self.train_label[index]
            if img.mode != "RGB":
                img = img.convert("RGB")
          
            r,g,b = self.img_split(img)
            entropy_r = self.calc_entropy_2dim(r)
            entropy_g = self.calc_entropy_2dim(g)
            entropy_b = self.calc_entropy_2dim(b)

            img = self.train_transform(img)

        else:
            img = Image.open(self.test_img_path[index])
            target = self.test_label[index]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = self.test_transform(img)

        return img, target, torch.tensor([entropy_r,entropy_g,entropy_b],requires_grad = False)

    def calc_entropy_1dim(self, gray_img):
        # 1-dim image entropy
        tmp = []
        for i in range(256):
            tmp.append(0)
        val = 0
        k = 0
        res = 0
        img = np.array(gray_img)
        for i in range(len(img)):
            for j in range(len(img[i])):
                val = img[i][j]
                tmp[val] = float(tmp[val] + 1)
                k = float(k + 1)
        for i in range(len(tmp)):
            tmp[i] = float(tmp[i] / k)
        for i in range(len(tmp)):
            if (tmp[i] == 0):
                res = res
            else:
                res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
        return res

    def calc_entropy_2dim(self, gray_img):
        gray_img = np.array(gray_img,dtype=np.float)
        gray_img = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(0)
        nn =gray_img.size()[2] * gray_img.size()[3]
        freq_matrix = np.zeros(shape=(256,256))

        avgpool = torch.nn.AvgPool2d(kernel_size=(3,3),stride=1,padding=1)
        avg_gray_img = avgpool(gray_img)

        gray_img = gray_img.squeeze(0).squeeze(0)
        avg_gray_img = avg_gray_img.squeeze(0).squeeze(0)
        gray_img = gray_img.view(1, -1).numpy()[0]
        avg_gray_img = avg_gray_img.view(1, -1).numpy()[0]

        for i,j in zip(gray_img,avg_gray_img):
            freq_matrix[int(i)][int(j)] += 1

        freq_p_matrix = np.reshape(freq_matrix,newshape=(1,-1))
        freq_p_matrix /= nn

        entropy = 0
        for p in freq_p_matrix[0]:
            if p == 0:
                continue
            else:
                entropy += (-1) * p * math.log(p,2.0)
        return entropy

    def img_split(self,img):
        img_arr = np.array(img,dtype=np.uint8)
        r_arr,g_arr,b_arr = img_arr[:,:,0],img_arr[:,:,1],img_arr[:,:,2]
        r,g,b = Image.fromarray(r_arr),Image.fromarray(g_arr),Image.fromarray(b_arr)
        return r,g,b

if __name__ == "__main__":
    dataSet = CUBDataset('D:\MasterDegree\Datasets\CUB_200_2011\CUB_200_2011',training=True)
    dataLoader = DataLoader(dataSet)
    for i,data in enumerate(dataLoader):
        print(data)
        break
