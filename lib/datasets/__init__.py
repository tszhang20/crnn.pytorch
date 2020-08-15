from .datasets import DataSet


def get_dataset(config):
    # 根据数据集名称调用相应的类
    if config.DATASET.DATASET == "OWN":
        return DataSet
    else:
        raise NotImplemented()
