import argparse
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils import get_dataloaders, build_model, load_ckpt, save_ckpt, accuracy

def train_one_epoch(model, loader, optim, device, criterion):
    model.train()
    pbar = tqdm(loader, desc="train", leave=False)
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
    ap.add_argument("--arch", default="resnet18", choices=["resnet18","resnet50","mobilenet_v3_small"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--resume", default=None, help="path to ckpt to resume weights from")
    ap.add_argument("--out", default="checkpoints/model.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    trainloader, testloader = get_dataloaders(batch_size=args.batch_size)

    model = build_model(args.arch).to(device)

    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")

        # pruned checkpoint format: {"model": nn.Module, ...}
        if isinstance(payload, dict) and "model" in payload:
            model = payload["model"].to(device)
        else:
            # baseline/old format: {"state_dict": ...}
            payload = load_ckpt(args.resume, device=device)
            model.load_state_dict(payload["state_dict"], strict=True)

    criterion = nn.CrossEntropyLoss()
    optim = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    sched = CosineAnnealingLR(optim, T_max=args.epochs)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, trainloader, optim, device, criterion)
        sched.step()
        acc = accuracy(model, testloader, device)
        best = max(best, acc)
        print(f"epoch={epoch:03d} loss={loss:.4f} acc={acc*100:.2f}% best={best*100:.2f}%")

    save_ckpt(args.out, model, args.arch, extra={"best_acc": best})
    print(f"saved: {args.out}")

if __name__ == "__main__":
    main()
