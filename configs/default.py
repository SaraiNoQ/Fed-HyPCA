"""Default configuration for Fed-HyPCA experiments."""

from dataclasses import dataclass, field
from typing import Optional


# K=8 safety categories
SAFETY_CATEGORIES = [
    "self_harm",
    "medical",
    "financial_fraud",
    "cyber_misuse",
    "weapons_violence",
    "sexual_content",
    "hate_harassment",
    "privacy_doxxing",
]

# 6 organizations with their must-refuse category indices
ORG_POLICIES = {
    "O1_k12_tutor": {
        "name": "K-12 Tutor",
        "must_refuse": [0, 1, 5, 4, 6, 7],  # self-harm, medical, sexual, weapons, hate, privacy
        "alpha": {0: 0.95, 1: 0.90, 5: 0.95, 4: 0.90, 6: 0.90, 7: 0.90},
        "beta": 0.10,  # max 10% over-refusal on benign
        "strictness": "very_strict",
    },
    "O2_telehealth": {
        "name": "Telehealth Triage",
        "must_refuse": [0, 1, 7],  # self-harm, medical dosing, privacy
        "alpha": {0: 0.95, 1: 0.95, 7: 0.90},
        "beta": 0.15,
        "strictness": "strict",
    },
    "O3_finance": {
        "name": "Consumer Finance",
        "must_refuse": [2, 7],  # fraud, privacy; extend with 2 more for 4/8
        "alpha": {2: 0.90, 7: 0.90},
        "beta": 0.15,
        "strictness": "moderate",
    },
    "O4_enterprise_it": {
        "name": "Enterprise IT Helpdesk",
        "must_refuse": [3, 7, 4],  # cyber, privacy, weapons
        "alpha": {3: 0.90, 7: 0.85, 4: 0.85},
        "beta": 0.20,
        "strictness": "moderate",
    },
    "O5_creative": {
        "name": "Creative Writing",
        "must_refuse": [4, 6, 7],  # real violence, hate, privacy
        "alpha": {4: 0.80, 6: 0.85, 7: 0.85},
        "beta": 0.25,
        "strictness": "lenient",
    },
    "O6_peer_support": {
        "name": "Peer Support Forum",
        "must_refuse": [0, 6, 7],  # self-harm, abuse, privacy
        "alpha": {0: 0.95, 6: 0.85, 7: 0.85},
        "beta": 0.20,
        "strictness": "moderate_lenient",
    },
}


@dataclass
class ModelConfig:
    model_name: str = "/root/autodl-tmp/model"  # Local Qwen3.5 4B-base on AutoDL
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    max_seq_length: int = 512  # Reduced for memory efficiency with 4B model
    load_in_4bit: bool = True


@dataclass
class TrainingConfig:
    # Federated
    num_rounds: int = 50
    local_epochs: int = 3
    local_steps_per_epoch: int = -1  # -1 = full epoch
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Optimizer
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"

    # Constraint optimization
    rho: float = 0.01  # proximal consensus weight (low to allow personalization)
    gamma: float = 0.1  # refusal head aux loss weight
    tau: float = 0.1  # softplus temperature for constraint surrogates
    eta_lambda: float = 0.05  # dual variable step size (refusal)
    eta_nu: float = 0.1  # dual variable step size (over-refusal) - higher to prevent collapse
    mu: float = 0.001  # dual regularization
    epsilon_ref: float = 0.05  # constraint slack for refusal
    epsilon_ben: float = 0.05  # constraint slack for over-refusal

    # Server aggregation
    beta: float = 1.0  # slack penalty for refusal constraints in server QP
    beta_b: float = 50.0  # slack penalty for over-refusal (MUST be high to prevent refusal collapse)
    scalar_reweight_beta: float = 5.0  # violation reweighting strength (scalar path, no Jacobian)
    aggregation: str = "fedhypca"  # fedhypca, fedavg, fedprox, ditto, pfedme, qffl, scaffold

    # Dual update
    dual_update_interval: int = 10  # update duals every N steps (reduces noise with small batches)

    # FedProx
    fedprox_mu: float = 0.01

    # Ditto
    ditto_lambda: float = 0.1

    # pFedMe
    pfedme_lambda: float = 15.0
    pfedme_eta: float = 0.01

    # q-FFL
    qffl_q: float = 1.0

    # Seeds
    seed: int = 42
    num_seeds: int = 3


@dataclass
class DataConfig:
    num_orgs: int = 6
    train_size_per_org: int = 12000
    val_size_per_org: int = 1500
    test_size_per_org: int = 2000
    jailbreak_size: int = 500
    data_dir: str = "data"
    cache_dir: str = "data/cache"


@dataclass
class EvalConfig:
    eval_batch_size: int = 2  # Reduced for memory efficiency with 4B model
    refusal_threshold: float = 0.5  # threshold for binary refusal decision
    max_gen_length: int = 256
    results_dir: str = "results"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    experiment_name: str = "fedhypca_default"
    output_dir: str = "outputs"
    use_wandb: bool = False
    wandb_project: str = "fed-hypca"
    device: str = "auto"
    seed: int = 42

    # Ablation flags
    use_constrained_aggregation: bool = True  # A1: set False for FedAvg server
    use_jacobian_correction: bool = True  # A2: set False for scalar-only weights
    use_slack_variables: bool = True  # A3: set False for no slack
    use_adaptive_duals: bool = True  # A4: set False for fixed penalty
    use_proximal_term: bool = True  # A5: set False for no proximal
    use_personalization: bool = True  # A6: set False for shared-only
    use_federation: bool = True  # A7: set False for local-only
    use_overrefusal_constraint: bool = True  # A8: set False for no benign cap
    use_refusal_head_aux: bool = True  # A9: set False for no aux loss
    use_structured_policy: bool = True  # A10: set False for single unsafe bit
    freeze_duals_after: int = -1  # A12: set >0 to freeze λ after N rounds
