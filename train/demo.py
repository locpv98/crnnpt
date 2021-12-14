import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
from torchvision.transforms import ToTensor
import models.crnn as crnn


model_path = '/home/ngoc/work/ocr/crnn.pytorch/netCRNN_100.pth'
# img_path = '/home/ngoc/work/ocr/crnn-license-plate-OCR/images/photo_2021-11-29_14-45-22.jpg'
img_path = '/home/ngoc/work/ocr/crnn.pytorch/valset/11A-652.75.jpg'
# img_path = '/home/ngoc/work/ocr/crnn.pytorch/images/56FCYikaub.jpg'
alphabet = '-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ncl = len(alphabet)+1 #39

model = crnn.CRNN(32, 3, ncl, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')), strict=False)
model.eval()

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((200, 32))
image = Image.open(img_path).convert('RGB')
# image = transformer(image)
if torch.cuda.is_available():
    image = ToTensor()(image).cuda()
image = image.view(1, *image.size())
image = Variable(image)

preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
