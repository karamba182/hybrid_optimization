import argparse, csv, os
import torch
from utils import build_model, load_ckpt, benchmark_latency_ms, file_size_mb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--log", default=None, help="append results to CSV")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")

    payload = load_ckpt(args.ckpt, device="cpu")
    arch = payload.get("arch", "resnet18")
    model = build_model(arch).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)

    lat_ms = benchmark_latency_ms(model, device=device, iters=args.iters, warmup=args.warmup)
    size_mb = file_size_mb(args.ckpt)
    print(f"latency_ms={lat_ms:.3f}  ckpt_size_mb={size_mb:.2f}")

    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        header = ["ckpt","arch","metric","value"]
        write_header = not os.path.exists(args.log)
        with open(args.log, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow([args.ckpt, arch, "latency_ms", f"{lat_ms:.6f}"])
            w.writerow([args.ckpt, arch, "ckpt_size_mb", f"{size_mb:.6f}"])

if __name__ == "__main__":
    main()
