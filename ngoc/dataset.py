from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor, Normalize
from PIL import Image
import torch

def resizePadding(img, chanel, width, height):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    desired_w, desired_h  = width, height #(width, height)
    img_w, img_h = img.size  # old_size[0] is in (width, height) format
    ratio = 1.0*img_w/img_h
    new_w = int(desired_h*ratio)
    new_w = new_w if desired_w == None else min(desired_w, new_w)
    img = img.resize((new_w, desired_h), Image.ANTIALIAS)

    # padding image
    if desired_w != None and desired_w > new_w:
        if chanel==3:
            new_img = Image.new("RGB", (desired_w, desired_h), color=255)
        else:
            new_img = Image.new("L",(desired_w, desired_h), color=255)
        new_img.paste(img, (0, 0))
        img = new_img
    
    img = ToTensor()(img)
    if chanel==3:
        img = Normalize(mean, std)(img)
    return img

def get_label_from_file(filename):
    """
        Create all strings by reading lines in specified files
    """
    strings = []
    
    with open(filename, "r", encoding="utf8") as f:
        lines = [l[0:200] for l in f.read().splitlines() if len(l) > 0]
        count = len(lines)
        if len(lines) == 0:
            raise Exception("No lines could be read in file")
        else:
            strings.extend(lines)
    return strings

def get_img_path_from_file(filename):
    """
        Create all strings by reading lines in specified files
    """
    import os
    p = os.path.dirname(filename)
    strings = []

    with open(filename, "r", encoding="utf8") as f:
        lines = [str(p)+'/'+l[0:200] for l in f.read().splitlines() if len(l) > 0]
        count = len(lines)
        if len(lines) == 0:
            raise Exception("No lines could be read in file")
        else:
            strings.extend(lines)
    return strings

class ImageDataset(Dataset):
    def __init__(self, list_img_path, list_label, chanel,transform=None):
        self.list_img = list_img_path
        self.list_label = list_label
        self.transform = transform
        self.chanel = chanel
    def __len__(self):
        return len(self.list_img)
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # print('dang den ',index)
        if self.chanel == 3:
            img = Image.open(self.list_img[index]).convert('RGB')
        else:
            img = Image.open(self.list_img[index]).convert('L')
        label = self.list_label[index]

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

class alignCollate(object):

    def __init__(self, chanel, imgW, imgH):
        self.chanel = chanel
        self.imgH = imgH
        self.imgW = imgW
    
    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        images = [resizePadding(image, self.chanel, self.imgW, self.imgH) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels

class DatasetLoader(object):
    def __init__(self, train_img_list, train_label_list, valid_img_list, valid_label_list, chanel, imgW, imgH):

        self.train_img_list = get_img_path_from_file(train_img_list)
        self.train_label_list = get_label_from_file(train_label_list)

        self.valid_img_list = get_img_path_from_file(valid_img_list)
        self.valid_label_list = get_label_from_file(valid_label_list)
        self.chanel = chanel
        self.imgW = imgW
        self.imgH = imgH

        self.train_dataset = ImageDataset( self.train_img_list, self.train_label_list, self.chanel) 
        self.test_dataset = ImageDataset(self.valid_img_list, self.valid_label_list, self.chanel)


    def train_loader(self, batch_size, num_workers=4, shuffle=True):

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=alignCollate(self.chanel,self.imgW, self.imgH)
        )

        return train_loader

    def test_loader(self, batch_size, num_workers=4, shuffle=True):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=alignCollate(self.chanel, self.imgW, self.imgH)
        )

        return test_loader

if __name__=='__main__':
    img_t = '/home/ngoc/work/ocr/crnn.pytorch/dataset/train.txt'
    lb_t = '/home/ngoc/work/ocr/crnn.pytorch/dataset/label.txt'
    img_v = '/home/ngoc/work/ocr/crnn.pytorch/valset/train.txt'
    lb_v = '/home/ngoc/work/ocr/crnn.pytorch/valset/valid.txt'
    import time
    from multiprocessing import cpu_count
    load = DatasetLoader(img_t,lb_t,img_v,lb_v, 512,32)
    for _ in range(100):
        train_loader = iter(load.train_loader(64, num_workers=cpu_count()))
        print(len(train_loader))
        i = 0
        while i < len(train_loader):
            start_time = time.time()
            X_train, y_train = next(train_loader)
            elapsed_time = time.time() - start_time
            i += 1
            print(i, elapsed_time, X_train.size())