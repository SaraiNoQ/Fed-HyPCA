"""Main entry point for Fed-HyPCA federated training.

Usage:
    # Full Fed-HyPCA
    python train_federated.py --aggregation fedhypca --num_rounds 50 --seed 42

    # FedAvg baseline
    python train_federated.py --aggregation fedavg --num_rounds 50 --seed 42

    # FedAvg + local dual constraints (ablation B7)
    python train_federated.py --aggregation fedavg_dual --num_rounds 50 --seed 42

    # Ablation A1: FedAvg server (no constrained aggregation)
    python train_federated.py --aggregation fedhypca --no_constrained_aggregation --seed 42

    # Ablation A6: shared-only (no personalization)
    python train_federated.py --aggregation fedhypca --no_personalization --seed 42

    # Local-only (no federation, ablation A7 / baseline B10)
    python train_federated.py --aggregation fedhypca --no_federation --seed 42
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.default import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig, EvalConfig
from src.utils.seed import set_seed
from src.models.lora_model import FedHyPCAModel, load_tokenizer
from src.data.dataset import build_benchmark
from src.data.taxonomy import get_all_org_ids
from src.federated.client import FedClient
from src.federated.server import FedServer
from src.evaluation.metrics import evaluate_all_orgs, save_results, print_results_table


def parse_args():
    parser = argparse.ArgumentParser(description="Fed-HyPCA Federated Training")

    # Model
    parser.add_argument("--model_name", type=str, default="/root/autodl-tmp/model")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true", default=False)

    # Federated training
    parser.add_argument("--aggregation", type=str, default="fedhypca",
                        choices=["fedhypca", "fedavg", "fedprox", "scaffold",
                                 "ditto", "pfedme", "qffl",
                                 "fedavg_dual", "ditto_dual"])
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--local_steps_per_epoch", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    # Constraint optimization
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--eta_lambda", type=float, default=0.05)
    parser.add_argument("--eta_nu", type=float, default=0.1)  # higher to prevent refusal collapse
    parser.add_argument("--mu", type=float, default=0.001)
    parser.add_argument("--epsilon_ref", type=float, default=0.05)
    parser.add_argument("--epsilon_ben", type=float, default=0.05)

    # Server aggregation
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta_b", type=float, default=50.0)  # high to prevent refusal collapse
    parser.add_argument("--scalar_reweight_beta", type=float, default=5.0)
    parser.add_argument("--dual_update_interval", type=int, default=10)

    # Baseline-specific
    parser.add_argument("--fedprox_mu", type=float, default=0.01)
    parser.add_argument("--ditto_lambda", type=float, default=0.1)
    parser.add_argument("--pfedme_lambda", type=float, default=15.0)
    parser.add_argument("--qffl_q", type=float, default=1.0)

    # Ablation flags
    parser.add_argument("--no_constrained_aggregation", action="store_true")  # A1
    parser.add_argument("--no_jacobian_correction", action="store_true")  # A2
    parser.add_argument("--no_slack_variables", action="store_true")  # A3
    parser.add_argument("--no_adaptive_duals", action="store_true")  # A4
    parser.add_argument("--no_proximal_term", action="store_true")  # A5
    parser.add_argument("--no_personalization", action="store_true")  # A6
    parser.add_argument("--no_federation", action="store_true")  # A7
    parser.add_argument("--no_overrefusal_constraint", action="store_true")  # A8
    parser.add_argument("--no_refusal_head_aux", action="store_true")  # A9
    parser.add_argument("--no_structured_policy", action="store_true")  # A10
    parser.add_argument("--freeze_duals_after", type=int, default=-1)  # A12

    # Data
    parser.add_argument("--num_orgs", type=int, default=6)
    parser.add_argument("--train_size_per_org", type=int, default=12000)
    parser.add_argument("--val_size_per_org", type=int, default=1500)
    parser.add_argument("--test_size_per_org", type=int, default=2000)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="data/cache")

    # Experiment
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="fed-hypca")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=25)

    return parser.parse_args()


def build_config(args) -> ExperimentConfig:
    """Build ExperimentConfig from argparse args."""
    model_config = ModelConfig(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit and not args.no_4bit,
    )

    training_config = TrainingConfig(
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        local_steps_per_epoch=args.local_steps_per_epoch,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        rho=args.rho,
        gamma=args.gamma,
        tau=args.tau,
        eta_lambda=args.eta_lambda,
        eta_nu=args.eta_nu,
        mu=args.mu,
        epsilon_ref=args.epsilon_ref,
        epsilon_ben=args.epsilon_ben,
        beta=args.beta,
        beta_b=args.beta_b,
        scalar_reweight_beta=args.scalar_reweight_beta,
        aggregation=args.aggregation,
        dual_update_interval=args.dual_update_interval,
        fedprox_mu=args.fedprox_mu,
        ditto_lambda=args.ditto_lambda,
        pfedme_lambda=args.pfedme_lambda,
        qffl_q=args.qffl_q,
        seed=args.seed,
    )

    data_config = DataConfig(
        num_orgs=args.num_orgs,
        train_size_per_org=args.train_size_per_org,
        val_size_per_org=args.val_size_per_org,
        test_size_per_org=args.test_size_per_org,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
    )

    eval_config = EvalConfig(results_dir=args.results_dir)

    exp_name = args.experiment_name or f"{args.aggregation}_r{args.num_rounds}_s{args.seed}"

    config = ExperimentConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        eval=eval_config,
        experiment_name=exp_name,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        seed=args.seed,
        use_constrained_aggregation=not args.no_constrained_aggregation,
        use_jacobian_correction=not args.no_jacobian_correction,
        use_slack_variables=not args.no_slack_variables,
        use_adaptive_duals=not args.no_adaptive_duals,
        use_proximal_term=not args.no_proximal_term,
        use_personalization=not args.no_personalization,
        use_federation=not args.no_federation,
        use_overrefusal_constraint=not args.no_overrefusal_constraint,
        use_refusal_head_aux=not args.no_refusal_head_aux,
        use_structured_policy=not args.no_structured_policy,
        freeze_duals_after=args.freeze_duals_after,
    )

    return config


def main():
    args = parse_args()
    config = build_config(args)

    # Set seed
    set_seed(config.seed)

    # Create output directories
    exp_dir = os.path.join(config.output_dir, config.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(config.eval.results_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {config.model.model_name}")
    tokenizer = load_tokenizer(config.model.model_name)

    # Build benchmark datasets
    print("Building benchmark datasets...")
    org_datasets = build_benchmark(
        data_config=config.data,
        tokenizer=tokenizer,
        seed=config.seed,
    )

    # Build model
    print(f"Loading model: {config.model.model_name}")
    model = FedHyPCAModel(config.model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.get_trainable_params()):,}")

    # Create clients
    org_ids = get_all_org_ids()[:config.data.num_orgs]
    clients = []
    for org_id in org_ids:
        client = FedClient(
            org_id=org_id,
            model=model,
            train_dataset=org_datasets[org_id]["train"],
            val_dataset=org_datasets[org_id]["val"],
            config=config,
            device=device,
        )
        clients.append(client)
    print(f"Created {len(clients)} clients: {[c.org_id for c in clients]}")

    # Handle local-only training (ablation A7 / baseline B10)
    if not config.use_federation:
        print("\n=== LOCAL-ONLY TRAINING (no federation) ===")
        client_states = {}
        for client in clients:
            print(f"\nTraining {client.org_id} locally...")
            local_state = client.local_train(
                global_state=model.get_lora_state_dict(),
                round_idx=0,
                aggregation_type=config.training.aggregation,
            )
            client_states[client.org_id] = local_state

        # Evaluate
        results = evaluate_all_orgs(
            model=model,
            org_datasets=org_datasets,
            config=config,
            device=device,
            global_state=None,
            client_states=client_states,
            split_name="test",
        )
        print_results_table(results)
        results_path = os.path.join(config.eval.results_dir, f"{config.experiment_name}.json")
        save_results(results, results_path)
        return

    # Create server and run federated training
    server = FedServer(config=config, model=model, clients=clients)

    print(f"\n=== FEDERATED TRAINING: {config.training.aggregation} ===")
    t_start = time.time()

    pbar = tqdm(range(config.training.num_rounds), desc="Training", unit="round")
    for r in pbar:
        round_start = time.time()
        round_metrics = server.run_round()
        round_time = time.time() - round_start
        avg_loss = round_metrics["avg_client_loss"]
        pbar.set_postfix(avg_loss=f"{avg_loss:.4f}", time=f"{round_time/60:.1f}min")

        # Periodic evaluation
        if (r + 1) % args.eval_every == 0 or r == config.training.num_rounds - 1:
            pbar.write(f"\n--- Evaluation at round {r + 1} ---")
            # Use last_local_state from each client (clients share model instance)
            client_states = {
                c.org_id: c.last_local_state for c in clients if c.last_local_state is not None
            }
            results = evaluate_all_orgs(
                model=model,
                org_datasets=org_datasets,
                config=config,
                device=device,
                global_state=server.global_state,
                client_states=client_states,
                split_name="val",
            )
            print_results_table(results)

            # Save intermediate results
            inter_path = os.path.join(
                config.eval.results_dir,
                f"{config.experiment_name}_round{r + 1}.json",
            )
            save_results(results, inter_path)

        # Periodic checkpoint
        if (r + 1) % args.save_every == 0:
            ckpt_path = os.path.join(exp_dir, f"checkpoint_round{r + 1}.pt")
            server.save_checkpoint(ckpt_path)

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time / 3600:.2f} hours")

    # Final evaluation on test set
    print("\n=== FINAL EVALUATION (test set) ===")
    # Use last_local_state from each client (clients share model instance)
    client_states = {
        c.org_id: c.last_local_state for c in clients if c.last_local_state is not None
    }
    final_results = evaluate_all_orgs(
        model=model,
        org_datasets=org_datasets,
        config=config,
        device=device,
        global_state=server.global_state,
        client_states=client_states,
        split_name="test",
    )
    print_results_table(final_results)

    # Save final results
    final_path = os.path.join(
        config.eval.results_dir, f"{config.experiment_name}_final.json"
    )
    save_results(final_results, final_path)

    # Save final checkpoint
    server.save_checkpoint(os.path.join(exp_dir, "checkpoint_final.pt"))

    # Save training history
    history_path = os.path.join(exp_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(server.history, f, indent=2, default=str)

    print(f"\nAll results saved to {config.eval.results_dir}/")
    print(f"Checkpoints saved to {exp_dir}/")


if __name__ == "__main__":
    main()
