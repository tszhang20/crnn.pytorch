import numpy as np
import time
import cv2
import os
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse


def parse_args():
    """参数解析"""
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, required=True, help='configuration filename', )
    ap.add_argument('--image_folder', type=str, required=True, help='the images to test')
    ap.add_argument('--checkpoint', type=str, required=True, help='the path to your checkpoints')
    arg = ap.parse_args()
    # 解析配置文件
    with open(arg.cfg, 'r') as f:
        conf = yaml.load(f)
        conf = edict(conf)
    # 字符库
    conf.DATASET.ALPHABETS = alphabets.alphabet
    conf.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    return conf, arg


def recognition(image_name, config, img, model, converter, device):
    h, w = img.shape
    # 第一步：将图像尺度固定为(32,x)，即按比例将高固定为32，并保证输入图像尺度与训练时一致
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h,
                     interpolation=cv2.INTER_CUBIC)
    h, w = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 3))
    # 图像预处理
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)
    img = img.to(device)
    img = img.view(1, *img.size())
    # 获得输出结果
    model.eval()
    preds = model(img)
    # 后处理
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # 写入结果
    with open('/home/lab/zts/datasets/warped_100_images_results.txt', 'a') as f:
        f.write(image_name + ' ' + sim_pred + '\n')
    f.close()
    print('results: {0}'.format(sim_pred))


if __name__ == '__main__':
    # 参数解析
    config, args = parse_args()
    # 加载模型
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = crnn.get_crnn(config, len(alphabets.alphabet)).to(device)
    print('loading pre trained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    # time
    started = time.time()
    # 标签转换
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    # 处理文件夹
    image_folder = args.image_folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        recognition(image_name, config, img, model, converter, device)
    # time
    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))
