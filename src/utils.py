import torch.nn as nn


def enable_dropout(model):
    """
    Enable dropout layers during test-time for MC Dropout inference.
    Call after model.eval() to keep batchnorm frozen but dropout active.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
            m.train()
