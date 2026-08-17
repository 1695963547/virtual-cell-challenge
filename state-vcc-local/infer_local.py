"""本地版推理脚本。

对应 Colab 教程里的:
    ! uv run state tx infer \\
      --output "competition/prediction.h5ad" \\
      --model-dir "competition/first_run" \\
      --checkpoint "competition/first_run/checkpoints/step=20000.ckpt" \\
      --adata "competition_support_set/competition_val_template.h5ad" \\
      --pert-col "target_gene"
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_SRC = ROOT / "state" / "src"
sys.path.insert(0, str(STATE_SRC))
os.chdir(ROOT / "state")
os.environ.setdefault("MPLBACKEND", "Agg")


def main() -> int:
    parser = argparse.ArgumentParser(description="STATE tx 推理本地启动器")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径(.ckpt)")
    parser.add_argument("--model-dir", type=str, default="competition/first_run", help="训练 run 目录(含 config.yaml)")
    parser.add_argument("--adata", type=str, default="competition_support_set/competition_val_template.h5ad")
    parser.add_argument("--output", type=str, default="competition/prediction.h5ad")
    parser.add_argument("--pert-col", type=str, default="target_gene")
    parser.add_argument("--embed-key", type=str, default=None,
                        help="obsm key for输入特征(X_state 等)。None=用 adata.X。SE 路线传 X_state")
    parser.add_argument("--celltype-col", type=str, default=None)
    parser.add_argument("--max-set-len", type=int, default=None, help="覆盖模型默认的 cell_set_len")
    args = parser.parse_args()

    # state tx infer 是 argparse,直接把参数塞进 sys.argv
    ckpt = args.checkpoint
    if not Path(ckpt).is_absolute():
        ckpt = str(ROOT / "state" / ckpt)

    cmd = [
        "state", "tx", "infer",
        "--checkpoint", ckpt,
        "--model-dir", args.model_dir,
        "--adata", args.adata,
        "--output", args.output,
        "--pert-col", args.pert_col,
    ]
    if args.celltype_col:
        cmd += ["--celltype-col", args.celltype_col]
    if args.max_set_len is not None:
        cmd += ["--max-set-len", str(args.max_set_len)]
    if args.embed_key is not None:
        cmd += ["--embed-key", args.embed_key]

    sys.argv = cmd
    print("[run]", " ".join(sys.argv))
    from state.__main__ import main as state_main  # noqa: E402

    state_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
