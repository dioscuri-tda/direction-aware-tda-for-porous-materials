from collections import OrderedDict

import torch
from TRIDCNNPyTorch.models import EfficientNet, DenseNet
from torch import nn

ALLOWED_MODELS = [
    "densenet-121",
    "densenet-169",
    "densenet-201",
    "densenet-264",
]


class TinyFullyConnected(torch.nn.Module):
    def __init__(self, p_dropout=0.3, fc_first_size=[100, 100, 50]):
        super(TinyFullyConnected, self).__init__()
        _layers = OrderedDict()
        for i in range(len(fc_first_size) - 1):
            _layers[f"lin{i}"] = torch.nn.Linear(fc_first_size[i], fc_first_size[i + 1])
            _layers[f"act{i}"] = torch.nn.ReLU()
            if i != len(fc_first_size) - 2:
                _layers[f"drop{i}"] = nn.Dropout(p=p_dropout)
        self.linear = nn.Sequential(_layers)

    def forward(self, x):
        x = self.linear(x)
        return x


class CNNTopoModel(nn.Module):
    def __init__(
        self,
        cnnmodel: nn.Module,
        fullyconnectedmodel: nn.Module,
        fc_second_size=[32, 32, 32],
        fc_last=50,
        p_dropout=0.3,
        n_targets=1,
    ):
        super(CNNTopoModel, self).__init__()
        # self.cnn = torch.nn.Sequential(*list(cnnmodel.children())[:-1])
        self.cnn = cnnmodel
        self.fullyconnected = fullyconnectedmodel
        self.n_layres = len(fc_second_size)
        _layers = OrderedDict()
        _layers["lin0"] = nn.Linear(self.cnn.num_features + fc_last, fc_second_size[0])
        _layers["act0"] = torch.nn.ReLU()
        if len(fc_second_size) > 1:
            _layers[f"drop0"] = nn.Dropout(p=p_dropout)
        for i in range(1, self.n_layres):
            _layers[f"lin{i}"] = torch.nn.Linear(fc_second_size[i - 1], fc_second_size[i])
            _layers[f"act{i}"] = torch.nn.ReLU()
            if i != len(fc_second_size) - 1:
                _layers[f"drop{i}"] = nn.Dropout(p=p_dropout)
        _layers["fc"] = torch.nn.Linear(fc_second_size[-1], n_targets)
        self.linear = nn.Sequential(_layers)
        pass

    def forward(self, structure, desc):
        x1 = self.cnn(structure).squeeze()
        x2 = self.fullyconnected(desc)

        if x1.dim() == 1:
            x1 = x1.reshape(1, -1)
        if x2.dim() == 1:
            x2 = x2.reshape(1, -1)

        x = torch.cat((x1, x2), dim=1)
        x = self.linear(x)
        return x


class FullyConnectedSecond(torch.nn.Module):
    def __init__(self, fullyconnectedmodel: nn.Module, fc_second_size=[32, 32, 32], fc_last=50, p_dropout=0.3, n_targets=1):
        super(FullyConnectedSecond, self).__init__()
        self.fullyconnected = fullyconnectedmodel
        _layers = OrderedDict()
        fc_sizes = [fc_last] + fc_second_size + [n_targets]
        for i in range(len(fc_sizes) - 1):
            _layers[f"lin{i}"] = torch.nn.Linear(fc_sizes[i], fc_sizes[i + 1])
            _layers[f"act{i}"] = torch.nn.ReLU()
            # if i != len(fc_sizes)-2:
            #     _layers[f'drop{i}'] = nn.Dropout(p=p_dropout)
        self.linear = nn.Sequential(_layers)

    def forward(self, structure, desc):
        x = self.fullyconnected(desc)
        x = self.linear(x)
        return x


def generate_model(
    model_name,
    use_descriptors=False,
    no_structure=False,
    fc_first_size=[100, 100, 50],
    fc_second_size=[32, 32, 32],
    p_dropout=0.3,
    n_targets=1,
):
    if use_descriptors and model_name in ["efficientnet-b0", "efficientnet-b4", "cnn"]:
        raise ValueError(f"Backbocne {model_name} not supported when descriptors used")
    if model_name == "efficientnet-b0":
        model = EfficientNet.EfficientNet3D.from_name(
            "efficientnet-b0", override_params={"num_classes": n_targets}, in_channels=1
        )
    elif model_name == "efficientnet-b4":
        model = EfficientNet.EfficientNet3D.from_name(
            "efficientnet-b4", override_params={"num_classes": n_targets}, in_channels=1
        )
    elif model_name == "densenet-121":
        model = DenseNet.generate_model(model_depth=121, num_classes=n_targets, use_fc_layer=not use_descriptors)
    elif model_name == "densenet-169":
        model = DenseNet.generate_model(model_depth=169, num_classes=n_targets, use_fc_layer=not use_descriptors)
    elif model_name == "densenet-201":
        model = DenseNet.generate_model(model_depth=201, num_classes=n_targets, use_fc_layer=not use_descriptors)
    elif model_name == "densenet-264":
        model = DenseNet.generate_model(model_depth=264, num_classes=n_targets, use_fc_layer=not use_descriptors)

    if use_descriptors and not no_structure:
        fcmodel = TinyFullyConnected(p_dropout=p_dropout, fc_first_size=fc_first_size)
        combinedmodel = CNNTopoModel(
            cnnmodel=model,
            fullyconnectedmodel=fcmodel,
            fc_second_size=fc_second_size,
            fc_last=fc_first_size[-1],
            p_dropout=p_dropout,
            n_targets=n_targets,
        )
        return combinedmodel
    elif use_descriptors and no_structure:
        fcmodel = TinyFullyConnected(p_dropout=p_dropout, fc_first_size=fc_first_size)
        combinedmodel = FullyConnectedSecond(
            fullyconnectedmodel=fcmodel, fc_second_size=fc_second_size, fc_last=fc_first_size[-1], p_dropout=p_dropout,
            n_targets=n_targets,
        )
        return combinedmodel
    else:
        return model
