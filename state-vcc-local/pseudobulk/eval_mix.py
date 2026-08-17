"""对 val 上的混合预测跑 cell-eval run（复用真值侧 DE），α 列表扫描。"""
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval_mix")

STATE = "/home/zjh/state-vcc-local/state"
PY = os.path.join(STATE, ".venv/bin/python")
TRUTH = "/home/zjh/vcc_official/validation/adata_Validation.h5ad"
REAL_DE = os.path.join(STATE, "competition/val_transfer_eval/real_de.csv")

ALPHAS = [0.25, 0.5, 0.75]

def run_eval(pred, out):
    cmd = [
        PY, "-c",
        "import logging,sys;logging.basicConfig(level=logging.INFO,"
        'format="%(asctime)s %(levelname)s %(name)s: %(message)s");'
        'sys.argv[0]="cell-eval";from cell_eval.__main__ import main;main()',
        "run",
        "-ap", pred, "-ar", TRUTH, "-dr", REAL_DE,
        "-o", out,
        "--pert-col", "target_gene", "--control-pert", "non-targeting",
        "--profile", "vcc", "--num-threads", "32",
    ]
    log.info("run: %s", " ".join(cmd))
    r = subprocess.run(cmd, cwd=STATE)
    return r.returncode == 0

for a in ALPHAS:
    pred = os.path.join(STATE, f"competition/prediction_val_mix_a{a}.h5ad")
    out = os.path.join(STATE, f"competition/val_transfer_eval/mix_a{a}")
    log.info("===== alpha=%s =====", a)
    if not run_eval(pred, out):
        log.error("FAIL alpha=%s", a)
        sys.exit(1)
log.info("all done")
