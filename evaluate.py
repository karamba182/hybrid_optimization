import argparse, csv, os
import torch
from utils import get_dataloaders, build_model, load_ckpt, accuracy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--int8", action="store_true", help="load ckpt that already contains an INT8-converted model")
    ap.add_argument("--log", default=None, help="append results to CSV")
    args = ap.parse_args()

    device = torch.device(args.device)
    _, testloader = get_dataloaders(batch_size=256)

    raw = torch.load(args.ckpt, map_location="cpu")

    # Format A: pruned checkpoint with full model
    if isinstance(raw, dict) and "model" in raw:
        model = raw["model"].to(device)
        arch = raw.get("arch", "pruned")
    else:
        # Format B: classic checkpoint with state_dict
        payload = load_ckpt(args.ckpt, device="cpu")
        arch = payload.get("arch", "resnet18")
        model = build_model(arch).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)

    model.eval()

    acc = accuracy(model, testloader, device)
    print(f"accuracy={acc*100:.2f}%")

    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        header = ["ckpt","arch","metric","value"]
        write_header = not os.path.exists(args.log)
        with open(args.log, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow([args.ckpt, arch, "accuracy", f"{acc:.6f}"])

if __name__ == "__main__":
    main()