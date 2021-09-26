import torch


def denormalize(img, stats):
    mean, std = torch.tensor(stats[0]).reshape(-1, 1, 1), torch.tensor(stats[1]).reshape(-1, 1, 1)
    img = (img * std) + mean
    return img