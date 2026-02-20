"""
DetectGPT classifier for PeerPrism baselines (self-contained; no external detect-gpt repo).
Log-likelihood under a causal LM; threshold -3.5 → human vs ai.
Supports local (e.g. gpt2-medium) or OpenAI (e.g. gpt-3.5-turbo-instruct).
"""
from typing import Any, Dict, Optional

try:
    import torch
    import numpy as np
    import transformers
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class DetectGPTClassifier:
    """Classify text as human or AI using log-likelihood under a causal LM (threshold -3.5)."""

    def __init__(
        self,
        base_model_name: str = "gpt2-medium",
        device: Optional[str] = None,
        openai_model: Optional[str] = None,
    ) -> None:
        self.openai_model = openai_model

        if openai_model is not None:
            import os
            if os.getenv("OPENAI_API_KEY") is None:
                raise ValueError(
                    "OPENAI_API_KEY required when using openai_model. "
                    "Set: export OPENAI_API_KEY=your_key"
                )
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("OpenAI required: pip install openai")
            self.base_model = None
            self.base_tokenizer = None
            self.device = None
            self.base_model_name = openai_model
            return

        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch and transformers required for local DetectGPT.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name

        base_model_kwargs: Dict[str, Any] = {}
        if "gpt-j" in base_model_name or "neox" in base_model_name:
            base_model_kwargs["torch_dtype"] = torch.float16
        if "gpt-j" in base_model_name:
            base_model_kwargs["revision"] = "float16"

        self.base_model_name = base_model_name
        try:
            self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
                base_model_name, **base_model_kwargs
            ).to(self.device)
        except Exception:
            self.base_model_name = "gpt2"
            self.base_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)

        optional_tok: Dict[str, Any] = {}
        if "facebook/opt-" in self.base_model_name:
            optional_tok["fast"] = False
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_model_name, **optional_tok
        )
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id

    def get_log_likelihood(self, text: str) -> float:
        """Mean log-likelihood of the text under the base model (or OpenAI)."""
        if self.openai_model is not None:
            return self._openai_log_likelihood(text)

        with torch.no_grad():
            tokenized = self.base_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)
            labels = tokenized.input_ids
            loss = self.base_model(**tokenized, labels=labels).loss.item()
            return -loss

    def _openai_log_likelihood(self, text: str) -> float:
        model_lower = self.openai_model.lower()
        is_instruct = "instruct" in model_lower
        is_chat = (
            any(x in model_lower for x in ["gpt-4", "gpt-3.5", "o1", "o3", "o4", "gpt-4o", "chatgpt"])
            and not is_instruct
        )
        if is_chat:
            raise ValueError(
                f"Chat model '{self.openai_model}' is not supported. "
                "Use gpt-3.5-turbo-instruct for DetectGPT."
            )
        max_chars = 4000
        prompt_text = text[:max_chars] if len(text) > max_chars else text
        response = self.openai_client.completions.create(
            model=self.openai_model,
            prompt=prompt_text,
            temperature=0,
            max_tokens=50,
            logprobs=5,
        )
        result = response.choices[0]
        if result.logprobs and result.logprobs.token_logprobs:
            logprobs_list = [lp for lp in result.logprobs.token_logprobs if lp is not None]
            if logprobs_list:
                return float(np.mean(logprobs_list))
        return -5.0

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Returns dict with predicted_label ("human" or "ai"), log_likelihood, threshold_used,
        and optionally rank, log_rank, entropy (local only).
        """
        threshold = -3.5
        try:
            log_likelihood = self.get_log_likelihood(text)
            predicted_label = "human" if log_likelihood >= threshold else "ai"
            results: Dict[str, Any] = {
                "predicted_label": predicted_label,
                "log_likelihood": float(log_likelihood),
                "threshold_used": float(threshold),
            }
            if self.base_model is not None:
                rank = self._get_rank(text, log=False)
                log_rank = self._get_rank(text, log=True) if rank is not None else None
                entropy = self._get_entropy(text)
                if rank is not None:
                    results["rank"] = float(rank)
                if log_rank is not None:
                    results["log_rank"] = float(log_rank)
                if entropy is not None:
                    results["entropy"] = float(entropy)
            return results
        except Exception as e:
            return {"predicted_label": "error", "error": str(e)}

    def _get_rank(self, text: str, log: bool = False) -> Optional[float]:
        if self.base_model is None:
            return None
        with torch.no_grad():
            tokenized = self.base_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)
            logits = self.base_model(**tokenized).logits[:, :-1]
            labels = tokenized.input_ids[:, 1:]
            matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
            if matches.shape[1] != 3:
                return None
            ranks, timesteps = matches[:, -1], matches[:, -2]
            if not (timesteps == torch.arange(len(timesteps), device=timesteps.device)).all():
                return None
            ranks = ranks.float() + 1
            if log:
                ranks = torch.log(ranks)
            return float(ranks.mean().item())

    def _get_entropy(self, text: str) -> Optional[float]:
        if self.base_model is None:
            return None
        import torch.nn.functional as F
        with torch.no_grad():
            tokenized = self.base_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)
            logits = self.base_model(**tokenized).logits[:, :-1]
            neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
            return float(-neg_entropy.sum(-1).mean().item())
