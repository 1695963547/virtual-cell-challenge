# STATE for Virtual Cell Challenge —— 本地版

把 [Colab 教程](https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l) 的流程搬到本地跑。

## 目录结构

```
state-vcc-local/
├── state/                       # 克隆下来的 STATE 仓库(完整代码)
├── download_data.py             # 数据下载脚本(带进度条和续传)
├── train_local.py               # 训练启动脚本(包装 state tx train)
├── infer_local.py               # 推理启动脚本(包装 state tx infer)
├── eval_local.py                # cell-eval 评估 + 打包 .vcc 提交文件
├── run_all.ps1                  # 一键串联整个流程(PowerShell 版)
├── competition_support_set.zip  # 数据集压缩包(下载后存在)
└── competition_support_set/     # 解压后的数据(运行 download+unzip 后才有)
    ├── competition_train.h5
    ├── competition_val_template.h5ad
    ├── k562_gwps.h5
    ├── rpe1.h5
    ├── jurkat.h5
    ├── k562.h5
    ├── hepg2.h5
    ├── starter.toml
    ├── gene_names.csv
    └── ESM2_pert_features.pt
```

## 环境要求

- Windows 10/11 + PowerShell 5.x 或 PowerShell 7
- [uv](https://docs.astral.sh/uv/) 0.11+  (Python 包管理,会自动装 Python 3.11)
- NVIDIA GPU(模型 650MB,4GB 显存勉强能跑,推荐 8GB+)
- CUDA 12.x 驱动(uv 会自动装匹配的 PyTorch)

直连网络就行,不用代理。Google Cloud Storage 在国内能直连,GitHub 也没问题。

## 完整流程(5 步)

### 1. 下载数据(约 1.7GB,15 分钟左右,看网速)

```powershell
cd e:\arc-virtual-cell-atlas-main\state-vcc-local
uv run --with requests python download_data.py
```

如果中途断了,再跑一次会自动续传(用 HTTP Range)。

下载完会得到 `competition_support_set.zip`,手动解压到当前目录(或者用 `Expand-Archive`):

```powershell
Expand-Archive -Path competition_support_set.zip -DestinationPath . -Force
```

### 2. 装依赖(首次约 5-10 分钟)

进 state 目录,uv 会自动读 `pyproject.toml` 装好所有依赖,并自动下载匹配的 PyTorch(带 CUDA):

```powershell
cd state
uv sync
```

> 如果只是想快速跑一次不想同步整个 venv,可以用 `uv run` 自动管理临时环境。

### 3. 训练模型

回到 state-vcc-local 根目录,跑训练脚本:

```powershell
cd ..
uv run --project state python train_local.py
```

> `uv run --project state python train_local.py --max-steps 500`

训练产物在 `competition/first_run/checkpoints/` 下,每 20000 步存一个。

### 4. 推理验证集

```powershell
uv run --project state python infer_local.py
```

输出 `competition/prediction.h5ad`,在 H1 验证扰动上的预测表达谱。

### 5. 评估 + 打包提交

```powershell
uv run --project state python eval_local.py
```

最终生成 `competition/prediction.prep.vcc`,这就是上传到排行榜的文件。

## 一键跑全流程

```powershell
.\run_all.ps1
```

这个脚本会按顺序执行:下载 → 解压 → 训练 → 推理 → 评估,中间任意一步挂了会停。



## 常见问题

**Q: 下载 `competition_support_set.zip` 总是被 reset?**
A: Google Cloud Storage 在某些网络下会断。`download_data.py` 走的是 HTTP Range 续传,断了再跑一次会接着下。也可以试试开代理。

**Q: 训练时报 "CUDA out of memory"?**
A: 调小 batch size 到 2 或 1,或者开 `precision=16-mixed`。

**Q: 想换数据,不用 Replogle 的辅助数据?**
A: 编辑 `competition_support_set/starter.toml`,删掉里面的 `k562_gwps / rpe1 / jurkat / k562 / hepg2` 行,只保留 `competition_train` 即可。

**Q: 想换基因特征(不用 ESM2)?**
A: 改 `data.kwargs.perturbation_features_file` 指向你自己的 `.pt` 文件,shape 应该是 `[num_perturbations, 5120]`(或别的固定维度,需要同步改 `model.kwargs.pert_dim`)。

## 参考链接

- STATE 仓库:https://github.com/ArcInstitute/state
- 比赛官网:https://virtualcellchallenge.org/
- STATE 预印本:https://www.biorxiv.org/content/10.1101/2025.06.26.661135
- cell-eval 工具:https://github.com/ArcInstitute/cell-eval
