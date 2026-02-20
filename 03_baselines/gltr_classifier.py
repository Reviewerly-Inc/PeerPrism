"""
GLTR (Giant Language Model Test Room) classifier for PeerPrism baselines.
Self-contained: uses one causal LM (e.g. GPT-2); no external GLTR repo.

Rule: if fraction of tokens in top-10 model predictions > threshold → "ai", else "human".
Default threshold = 2/3.
"""
from typing import Any, Dict, Optional

try:
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class GLTRClassifier:
    """Classify text as human or AI using token-rank (top-10 ratio) under a causal LM."""

    def __init__(self, model_id: str = "gpt2-medium", device: Optional[str] = None) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch and transformers required for GLTR.")
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        self.model.eval()
        bos = self.tokenizer.bos_token or self.tokenizer.eos_token
        start_ids = self.tokenizer(bos, return_tensors="pt").input_ids[0].to(self.device)
        self.start_token = start_ids

    def _check_probabilities(self, text: str) -> Dict[str, Any]:
        """Return real_topk: list of (rank, prob) for each token (rank in model's top-k)."""
        max_len = 1022
        tok = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        token_ids = tok.input_ids[0].to(self.device)
        token_ids = torch.cat([self.start_token, token_ids])
        if token_ids.shape[0] > 1024:
            token_ids = token_ids[:1024]
        vocab_size = self.model.config.vocab_size
        token_ids = torch.clamp(token_ids, 0, vocab_size - 1)
        with torch.no_grad():
            out = self.model(token_ids.unsqueeze(0))
            all_logits = out.logits[0, :-1]
        if all_logits.dim() == 1:
            all_logits = all_logits.unsqueeze(0)
        all_probs = torch.softmax(all_logits, dim=1)
        y = token_ids[1:]
        y = torch.clamp(y, 0, vocab_size - 1)
        seq_len = min(y.shape[0], all_probs.shape[0])
        y = y[:seq_len]
        all_probs = all_probs[:seq_len]
        if seq_len == 0:
            return {"real_topk": []}
        sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
        real_topk = []
        for i in range(seq_len):
            try:
                pos = int(np.where(sorted_preds[i] == y[i].item())[0][0])
                prob = float(all_probs[i, y[i].item()].item())
                real_topk.append((pos, prob))
            except (IndexError, ValueError):
                real_topk.append((1000, 0.0))
        return {"real_topk": real_topk}

    def classify(self, text: str, threshold: float = 2 / 3) -> Dict[str, Any]:
        """Return predicted_label (human/ai), top10_ratio, top10_count, top100_count, top1000_count, other_count, total_tokens, threshold_used."""
        try:
            payload = self._check_probabilities(text)
            real_topk = payload.get("real_topk") or []
            total_tokens = len(real_topk)
            if total_tokens == 0:
                return {"predicted_label": "error", "error": "No valid tokens"}
            count = {"top_10": 0, "top_100": 0, "top_1000": 0, "other": 0}
            for rank, _ in real_topk:
                if 0 <= rank < 10:
                    count["top_10"] += 1
                elif rank < 100:
                    count["top_100"] += 1
                elif rank < 1000:
                    count["top_1000"] += 1
                else:
                    count["other"] += 1
            top10_ratio = count["top_10"] / total_tokens
            predicted_label = "ai" if top10_ratio > threshold else "human"
            return {
                "predicted_label": predicted_label,
                "top10_ratio": round(top10_ratio, 4),
                "top10_count": count["top_10"],
                "top100_count": count["top_100"],
                "top1000_count": count["top_1000"],
                "other_count": count["other"],
                "total_tokens": total_tokens,
                "threshold_used": threshold,
            }
        except Exception as e:
            return {"predicted_label": "error", "error": str(e)}
