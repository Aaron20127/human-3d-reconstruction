
import torch.nn as nn


class BaseNet_1(nn.Module):
    def __init__(self):
        super(BaseNet_1, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }



class BaseNet_2(nn.Module):
    def __init__(self):
        super(BaseNet_2, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }


class BaseNet_3(nn.Module):
    def __init__(self):
        super(BaseNet_3, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }


class BaseNet_4(nn.Module):
    def __init__(self):
        super(BaseNet_4, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }


class BaseNet_5(nn.Module):
    def __init__(self):
        super(BaseNet_5, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }


class BaseNet_6(nn.Module):
    def __init__(self):
        super(BaseNet_6, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }


class BaseNet_7(nn.Module):
    def __init__(self):
        super(BaseNet_7, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }

class BaseNet_8(nn.Module):
    def __init__(self):
        super(BaseNet_8, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }


class BaseNet_9(nn.Module):
    def __init__(self):
        super(BaseNet_9, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 4096+1024),
            nn.ReLU(),
            nn.Linear(4096+1024, 4096+1024),
            nn.ReLU(),
            nn.Linear(4096+1024, 4096+1024),
            nn.ReLU(),
            nn.Linear(4096+1024, 4096+1024),
            nn.ReLU(),
            nn.Linear(4096+1024, 4096+1024),
            nn.ReLU(),
            nn.Linear(4096+1024, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }


class BaseNet_10(nn.Module):
    def __init__(self):
        super(BaseNet_10, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 4096+2048),
            nn.ReLU(),
            nn.Linear(4096+2048, 4096+2048),
            nn.ReLU(),
            nn.Linear(4096+2048, 4096+2048),
            nn.ReLU(),
            nn.Linear(4096+2048, 4096+2048),
            nn.ReLU(),
            nn.Linear(4096+2048, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }


class BaseNet_11(nn.Module):
    def __init__(self):
        super(BaseNet_11, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 4096+2048),
            nn.ReLU(),
            nn.Linear(4096+2048, 4096+2048),
            nn.ReLU(),
            nn.Linear(4096+2048, 4096+2048),
            nn.ReLU(),
            nn.Linear(4096+2048, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }


class BaseNet_12(nn.Module):
    def __init__(self):
        super(BaseNet_12, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 4096+4096),
            nn.ReLU(),
            nn.Linear(4096+4096, 4096+4096),
            nn.ReLU(),
            nn.Linear(4096+4096, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }


BaseNet = {
    'BaseNet_1': BaseNet_1,
    'BaseNet_2': BaseNet_2,
    'BaseNet_3': BaseNet_3,
    'BaseNet_4': BaseNet_4,
    'BaseNet_5': BaseNet_5,
    'BaseNet_6': BaseNet_6,
    'BaseNet_7': BaseNet_7,
    'BaseNet_8': BaseNet_8,
    'BaseNet_9': BaseNet_9,
    'BaseNet_10': BaseNet_10,
    'BaseNet_11': BaseNet_11,
    'BaseNet_12': BaseNet_12,
}