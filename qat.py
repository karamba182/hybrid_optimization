import argparse
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils import get_dataloaders, build_model, load_ckpt, save_ckpt, accuracy

def train_one_epoch(model, loader, optim, device, criterion):
    model.train()
    pbar = tqdm(loader, desc="qat-train", leave=False)
    total_loss = 0.0
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / max(1, (pbar.n + 1)))
    return total_loss / max(1, len(loader))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--out", required=True, help="output ckpt (INT8 converted weights saved into float model structure)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu", help="QAT/INT8 is best reproduced on CPU")
    args = ap.parse_args()

    device = torch.device(args.device)
    trainloader, testloader = get_dataloaders(batch_size=args.batch_size)

    payload = load_ckpt(args.in_ckpt, device="cpu")
    arch = payload.get("arch","resnet18")

    model = build_model(arch).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)

    # QAT config (CPU)
    model.train()
    model.fuse_model = getattr(model, "fuse_model", None)  # keep compatible with other models
    backend = "fbgemm"
    torch.backends.quantized.engine = backend
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    torch.ao.quantization.prepare_qat(model, inplace=True)

    criterion = nn.CrossEntropyLoss()
    optim = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    sched = CosineAnnealingLR(optim, T_max=args.epochs)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, trainloader, optim, device, criterion)
        sched.step()
        acc = accuracy(model, testloader, device)
        best = max(best, acc)
        print(f"qat_epoch={epoch:03d} loss={loss:.4f} acc={acc*100:.2f}% best={best*100:.2f}%")

    # Convert to int8
    model.eval()
    int8_model = torch.ao.quantization.convert(model, inplace=False)

    # Save state_dict (contains quantized weights modules)
    save_ckpt(args.out, int8_model, arch, extra={"best_acc": best, "quantized": True, "backend": backend})
    print(f"saved INT8: {args.out}")

if __name__ == "__main__":
    main()
