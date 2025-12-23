import os, time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def get_dataloaders(batch_size=128, num_workers=0, data_dir="./data"):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    testset  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
    pin = torch.cuda.is_available()
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return trainloader, testloader

def build_model(arch="resnet18", num_classes=10):
    arch = arch.lower()
    if arch == "resnet18":
        model = torchvision.models.resnet18(weights=None)
    elif arch == "resnet50":
        model = torchvision.models.resnet50(weights=None)
    elif arch == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small(weights=None)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    # CIFAR-10: replace stem for 32x32
    if hasattr(model, "conv1"):
        model.conv1 = nn.Conv2d(3, model.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    # classifier head
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        # mobilenet v3
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

def save_ckpt(path, model, arch, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"arch": arch, "state_dict": model.state_dict()}
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)

def load_ckpt(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    return payload

def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def benchmark_latency_ms(model, device, input_shape=(1,3,32,32), iters=200, warmup=50):
    model.eval()
    x = torch.randn(*input_shape, device=device)
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    # timed
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000 / iters
