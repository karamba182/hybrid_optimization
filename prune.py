import argparse
import torch
import torch.nn as nn
import torch_pruning as tp
from utils import build_model, load_ckpt, save_ckpt

def collect_prunable_convs(model):
    # prune conv layers except the very first stem conv
    convs = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            convs.append((name, m))
    # drop first conv if present
    if convs:
        convs = convs[1:]
    return convs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prune_ratio", type=float, default=0.6, help="fraction of channels to prune (approx)")
    ap.add_argument("--arch", default=None, help="override arch")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    payload = load_ckpt(args.in_ckpt, device="cpu")
    arch = args.arch or payload.get("arch","resnet18")

    model = build_model(arch).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    example_inputs = torch.randn(1,3,32,32, device=device)
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

    convs = collect_prunable_convs(model)
    if not convs:
        raise RuntimeError("No prunable conv layers found.")

    # Simple global ranking by L1 norm per output channel across all convs.
    scores = []
    for name, conv in convs:
        w = conv.weight.detach()
        # per-out-channel L1 norm
        s = w.abs().sum(dim=(1,2,3))
        for i, val in enumerate(s.tolist()):
            scores.append((val, name, i))

    scores.sort(key=lambda x: x[0])  # smallest => prune first
    k = int(len(scores) * args.prune_ratio)
    to_prune = scores[:k]

    # group indices per layer
    layer_to_idxs = {}
    name_to_module = {n:m for n,m in convs}
    for _, lname, idx in to_prune:
        layer_to_idxs.setdefault(lname, []).append(idx)

    total_pruned = 0
    for lname, idxs in layer_to_idxs.items():
        conv = name_to_module[lname]
        idxs = sorted(set(idxs))
        # safety: do not prune all channels
        if len(idxs) >= conv.out_channels:
            idxs = idxs[: max(0, conv.out_channels - 1)]
        if not idxs:
            continue
        plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channels, idxs=idxs)
        plan.exec()
        total_pruned += len(idxs)

    save_ckpt(args.out, model, arch, extra={"prune_ratio": args.prune_ratio, "channels_pruned": total_pruned})
    print(f"Pruned channels: {total_pruned}  Saved: {args.out}")

if __name__ == "__main__":
    main()
