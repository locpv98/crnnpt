import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    # print(imageBin)
    # img = cv2.imread(imageBin)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(str(k).encode(), str(v).encode())


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = str(imagePathList[i])
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        # cache[labelKey] = label
        cache[labelKey] = label.encode(encoding='UTF-8',errors='strict')
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

def create_strings_from_file(filename):
    """
        Create all strings by reading lines in specified files
    """

    strings = []

    with open(filename, "r", encoding="utf8") as f:
        lines = [l[0:200] for l in f.read().splitlines() if len(l) > 0]
        if len(lines) == 0:
            raise Exception("No lines could be read in file")
        else:
            strings.extend(lines)
    return strings

def create_train_from_file(filename):
    """
        Create all strings by reading lines in specified files
    """
    import os
    p = os.path.dirname(filename)
    strings = []

    with open(filename, "r", encoding="utf8") as f:
        lines = [str(p)+'/'+l[0:200] for l in f.read().splitlines() if len(l) > 0]
        if len(lines) == 0:
            raise Exception("No lines could be read in file")
        else:
            strings.extend(lines)
    return strings


if __name__ == '__main__':
    train_list = create_train_from_file('/home/ngoc/work/ocr/crnn.pytorch/valset/train.txt')
    lb_list = create_strings_from_file('/home/ngoc/work/ocr/crnn.pytorch/valset/valid.txt')
    
    createDataset('/home/ngoc/work/ocr/crnn.pytorch/data/train', train_list, lb_list)