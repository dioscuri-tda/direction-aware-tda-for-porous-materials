import torch


def get_optimizer(model, optimizer_type: str, lr: float):
    if optimizer_type == "SGD":
        print("Using SGD as optimizer")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        print("Using Adam as optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer
