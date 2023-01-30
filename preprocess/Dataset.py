import sys

import numpy as np
import torch
import torch.utils.data

from model import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        self.data = data
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = 9

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst)) if len(inst) < max_len else inst[:max_len]
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, event_type = list(zip(*insts))
    time = pad_time(time)
    return time, event_type


def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
