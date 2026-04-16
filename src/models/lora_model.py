"""LoRA model wrapper for Fed-HyPCA.

Wraps a base LLM with LoRA adapters and a refusal head.
Supports extracting/loading LoRA parameters for federated communication.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

from src.models.refusal_head import RefusalHead


class FedHyPCAModel(nn.Module):
    """Base LLM + LoRA adapters + RefusalHead.

    The base LLM is frozen. Only LoRA params and refusal head are trainable.
    """

    def __init__(self, model_config):
        super().__init__()
        self.config = model_config

        # Load base model with quantization
        bnb_config = None
        if model_config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Enable gradient checkpointing to save memory
        self.base_model.gradient_checkpointing_enable()

        # Apply LoRA
        lora_config = LoraConfig(
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            target_modules=model_config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.base_model = get_peft_model(self.base_model, lora_config)

        # Refusal head
        hidden_dim = self.base_model.config.hidden_size
        self.refusal_head = RefusalHead(hidden_dim)
        # Move refusal head to same device as model
        device = next(self.base_model.parameters()).device
        self.refusal_head = self.refusal_head.to(device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass returning both LM outputs and refusal scores."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        # Get hidden states for refusal head
        hidden_states = outputs.hidden_states[-1]  # last layer
        refusal_scores = self.refusal_head(hidden_states, attention_mask)

        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
            "refusal_scores": refusal_scores,
            "hidden_states": hidden_states,
        }

    def get_lora_state_dict(self) -> OrderedDict:
        """Extract only LoRA parameters + refusal head parameters."""
        state = OrderedDict()
        for name, param in self.base_model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                state[name] = param.data.clone()
        for name, param in self.refusal_head.named_parameters():
            state[f"refusal_head.{name}"] = param.data.clone()
        return state

    def set_lora_state_dict(self, state_dict: OrderedDict):
        """Load LoRA parameters + refusal head parameters."""
        model_state = dict(self.base_model.named_parameters())
        for name, param in state_dict.items():
            if name.startswith("refusal_head."):
                rh_name = name[len("refusal_head."):]
                rh_state = dict(self.refusal_head.named_parameters())
                if rh_name in rh_state:
                    rh_state[rh_name].data.copy_(param)
            elif name in model_state:
                model_state[name].data.copy_(param)

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        """Get all trainable parameters (LoRA + refusal head)."""
        params = []
        for param in self.base_model.parameters():
            if param.requires_grad:
                params.append(param)
        for param in self.refusal_head.parameters():
            params.append(param)
        return params

    def get_lora_param_names(self) -> list[str]:
        """Get names of all LoRA + refusal head parameters."""
        names = []
        for name, param in self.base_model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                names.append(name)
        for name in self.refusal_head.state_dict():
            names.append(f"refusal_head.{name}")
        return names


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
