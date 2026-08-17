"""探针：复现训练挂死点。按 valsplit TOML 建 datamodule，逐个验证：
1. setup() 切分
2. val_dataloader() 建 loader
3. num_workers=4 拉一个 val batch
4. num_workers=4 拉一个 train batch
每步打印时间戳，挂在哪里一目了然。
"""
import os
import sys
import time

STATE = "/home/zjh/state-vcc-local/state"
sys.path.insert(0, STATE + "/src")
os.chdir(STATE)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("CELL_LOAD_H5_RDCC_NBYTES", "1073741824")

def ts():
    return time.strftime("%T")

from cell_load.data_modules.perturbation_dataloader import PerturbationDataModule  # noqa: E402

print(ts(), "实例化 DataModule")
dm = PerturbationDataModule(
    toml_config_path="vci_pretrain/starter_valsplit.toml",
    embed_key="X_state",
    output_space="all",
    pert_rep="onehot",
    basal_rep="sample",
    num_workers=4,
    pin_memory=False,
    n_basal_samples=1,
    basal_mapping_strategy="random",
    should_yield_control_cells=True,
    batch_col="batch_var",
    pert_col="target_gene",
    cell_type_key="cell_type",
    control_pert="non-targeting",
    map_controls=True,
    perturbation_features_file="competition_support_set/ESM2_pert_features.pt",
    barcode=True,
    batch_size=8,
    cell_sentence_len=512,
)
print(ts(), "setup()")
dm.setup()
print(ts(), f"splits train/val/test = {len(dm.train_datasets)}/{len(dm.val_datasets)}/{len(dm.test_datasets)}")

print(ts(), "创建 val_dataloader")
vdl = dm.val_dataloader()
print(ts(), "val_dataloader 返回:", type(vdl), "len:", len(vdl) if hasattr(vdl, "__len__") else "?")

print(ts(), "拉第一个 val batch（num_workers=4）...")
t0 = time.time()
for b in vdl:
    print(ts(), f"val batch OK, {time.time()-t0:.1f}s, keys:", sorted(b.keys())[:10])
    break

print(ts(), "拉第一个 train batch...")
tdl = dm.train_dataloader()
t0 = time.time()
for b in tdl:
    print(ts(), f"train batch OK, {time.time()-t0:.1f}s, keys:", sorted(b.keys())[:10])
    break

print(ts(), "探针完成")
