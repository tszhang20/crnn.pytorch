import argparse
from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.datasets import get_dataset
from lib.core import function
import lib.config.alphabets as alphabets
from lib.utils.utils import model_info

from tensorboardX import SummaryWriter
# from torch_baidu_ctc import CTCLoss, PyTorch1.2以下版本
from torch.nn import CTCLoss


def parse_arg():
    # 训练时必须指定.yaml配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    # 加载词库
    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    # 返回配置信息
    return config


def main():
    # 配置信息
    config = parse_arg()
    # 字典形式返回日志文件路径{'chs_dir': xx, 'tb_dir': xx}
    output_dict = utils.create_log_folder(config)
    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    # 写值
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    # 建立模型
    model = crnn.get_crnn(config, len(alphabets.alphabet))
    # gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")
    model = model.to(device)
    # 损失函数
    criterion = CTCLoss()
    # epoch数
    last_epoch = config.TRAIN.BEGIN_EPOCH
    # 优化器
    optimizer = utils.get_optimizer(config, model)
    # 学习率
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    # 是否加载预训练模型
    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        model.cnn.load_state_dict(model_dict)
        if config.TRAIN.FINETUNE.FREEZE:
            for p in model.cnn.parameters():
                p.requires_grad = False
    # 是否重新训练
    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)
    # 打印模型信息
    model_info(model)
    # 训练数据集
    train_dataset = get_dataset(config)(config, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    # 验证数据集
    val_dataset = get_dataset(config)(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    best_acc = 0.5
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        # 开始训练
        function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch,
                       writer_dict)
        # 学习率更新规则
        lr_scheduler.step()
        # 准确率(验证集)
        acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch,
                                writer_dict)
        # 保存最佳模型
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        print("is best:", is_best)
        print("best acc is:", best_acc)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_acc": best_acc,
            }, os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )
    # 关闭文件
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
