import time
from pathlib import Path

import torch
import torch.optim as optim


def get_optimizer(config, model):
    optimizer = None
    # 优化器
    if config.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV
        )
    elif config.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
        )
    elif config.TRAIN.OPTIMIZER == "rmsprop":
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
        )
    return optimizer


def create_log_folder(cfg):
    # 创建输出文件夹
    root_output_dir = Path(cfg.OUTPUT_DIR)
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    # 日志文件名称
    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    checkpoints_output_dir = root_output_dir / dataset / model / time_str / 'checkpoints'
    print('=> creating {}'.format(checkpoints_output_dir))
    checkpoints_output_dir.mkdir(parents=True, exist_ok=True)
    # tensorboard日志文件
    tensorboard_log_dir = root_output_dir / dataset / model / time_str / 'log'
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    # 返回日志文件路径
    return {'chs_dir': str(checkpoints_output_dir), 'tb_dir': str(tensorboard_log_dir)}


def get_batch_label(d, i):
    label = []
    for idx in i:
        label.append(list(d.labels[idx].values())[0])
    return label


class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=False):
        # 是否忽略大小写
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'
        self.dict = {}
        for i, char in enumerate(alphabet):
            # 索引0表示空白字符
            self.dict[char] = i + 1

    def encode(self, text):
        # 编码
        length = []
        result = []
        decode_flag = True if type(text[0]) == bytes else False
        for item in text:
            if decode_flag:
                item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length, raw=False):
        # 解码，处理单个字符的情况
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        # 处理多个字符
        else:
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


def model_info(model):
    # 参数量
    n_p = sum(x.numel() for x in model.parameters())
    # 可训练参数量
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))
