import numpy as np
import torch


class DownSample(torch.nn.Module):
    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        if self.scale_factor == 1:
            return x[0, :, :, :, :]
        x = torch.nn.Upsample(scale_factor=self.scale_factor, mode="trilinear")(x)
        x = x[0, :, :, :, :]
        return x


class RandomRoll(torch.nn.Module):
    def __init__(self, ratio_y: float = 0, ratio_z: float = 0):
        super().__init__()
        self.ratio_y = ratio_y / 2.0
        self.ratio_z = ratio_z / 2.0

    def forward(self, x):
        if self.ratio_y == 0 and self.ratio_z == 0:
            return x
        _, _, size_y, size_z = x.shape
        size_y = int(size_y * self.ratio_y)
        size_z = int(size_z * self.ratio_z)
        rand_roll_y = np.random.randint(low=-size_y, high=size_y)
        rand_roll_z = np.random.randint(low=-size_z, high=size_z)
        x = torch.roll(x, rand_roll_y, 2)
        x = torch.roll(x, rand_roll_z, 3)
        return x


class RandomFlip(torch.nn.Module):
    def __init__(self, p_flip: float):
        super().__init__()
        self.p_flip = p_flip

    def forward(self, x):
        if self.p_flip < 1e-5:
            return x
        if np.random.random() < self.p_flip:
            x = torch.flip(x, [1])
        if np.random.random() < self.p_flip:
            x = torch.flip(x, [2])
        if np.random.random() < self.p_flip:
            x = torch.flip(x, [3])
        return x


class EarlyStopper:
    def __init__(self, model_path: str, patience=5, min_delta=0):
        self.model_path = model_path
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model.state_dict(), self.model_path)
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
