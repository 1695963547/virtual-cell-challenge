
from dataclasses import dataclass
import os
@dataclass
class FlowConfig:
    # Flow model type
    model_type: str = 'hierarchical'

    # Flow Matching specific parameters
    batch_size: int = 32
    ntoken: int = 512
    d_model: int = 512
    lr: float = 1e-5
    steps: int = 5000
    eta_min: float = 1e-7
    devices: str = "1"
    test_only: bool = False
    # Perturbation related parameters
    data_name: str = "combosciplex"
    perturbation_function: str = 'crisper' 
    noise_type: str = "Gaussian"
    poisson_alpha: float = 0.8
    poisson_target_sum: int = -1

    print_every: int = 5000
    mode: str = 'predict_y' # predict_y, predict_p
    result_path: str = './result'
    perturbation_fusion_method: str = 'sum' # mlp, sum
    fusion_method: str = 'cross' # cross , concat, add
    infer_top_gene: int = 1000
    n_top_genes: int = 5000
    checkpoint_path: str = ''
    gamma: float = 0.0
    split_method: str = 'additive'
    use_mmd_loss: bool = False
    fold: int = 0
    use_negative_edge: bool = False
    topk: int = 15
    num_workers: int = 8
    eval_n_cells: int = 128
    eval_ode_steps: int = 10  # VCC: 10 步足够（rk4 每步 4 次前向，20 步耗时翻倍收益小）
    eval_bf16: bool = False  # VCC: eval 时模型转 bf16 + flash attention 加速
    eval_subset_perturbations: int = 0  # VCC: >0 时监控 eval 只用等间隔采样的 N 个扰动（0=全部）
    seed: int = 0  # VCC: >0 时固定随机种子（DDP 下每 rank 自动错开）
    
    def __post_init__(self):
        if self.data_name == 'norman_umi_go_filtered':
            self.n_top_genes = 5054
            self.ntoken = 6000
        if self.data_name == 'norman':
            self.n_top_genes = 5000
            self.ntoken = 6000
        if self.data_name == 'vcc':
            self.n_top_genes = 18080
            self.ntoken = 18084  # 18080 genes + 4 special tokens
        path = self.make_path()

    def make_path(self):
        exp_name = '-'.join(['flow', 
                             f'fusion_{self.fusion_method}',
                            f'{self.data_name}', 
                            self.model_type, 
                            self.mode, 
                            f'gamma_{self.gamma}',
                            f'perturbation_function_{self.perturbation_function}',
                            f'lr_{self.lr}', 
                            f'dim_model_{self.d_model}', 
                            f'infer_top_gene_{self.infer_top_gene}',
                            f'split_method_{self.split_method}',
                            f'use_mmd_loss_{self.use_mmd_loss}',
                            f'fold_{self.fold}',
                            f'use_negative_edge_{self.use_negative_edge}',
                            f'topk_{self.topk}',
                            ])
        return os.path.join(self.result_path, exp_name)