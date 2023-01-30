import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self):
        super().__init__()


    def forward(self, event_type, event_time, non_pad_mask):
        pass
