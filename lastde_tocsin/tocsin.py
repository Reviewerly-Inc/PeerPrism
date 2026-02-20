# PeerPrism copy: Lastde++ (Tocsin) scoring. Uses HuggingFace when /pretrain_models is missing.
import os
import sys
import random
import numpy as np
import torch
import tqdm
import json
import math
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCORING_DIR = os.path.join(SCRIPT_DIR, "scoring_methods")
if SCORING_DIR not in sys.path:
    sys.path.insert(0, SCORING_DIR)
import fastMDE  # type: ignore
import bart_score  # type: ignore

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_fullnames = {
    'gptj_6b': 'gpt-j-6b', 'gptneo_2.7b': 'gpt-neo-2.7B', 'gpt2_xl': 'gpt2-xl',
    'opt_2.7b': 'opt-2.7b', 'bloom_7b': 'bloom-7b1', 'falcon_7b': 'falcon-7b',
    'gemma_7b': "gemma-7b", 'llama1_13b': 'Llama-13b', 'llama2_13b': 'Llama-2-13B-fp16',
    'llama3_8b': 'Llama-3-8B', 'opt_13b': 'opt-13b', 'phi2': 'phi-2', "mgpt": 'mGPT',
    'qwen1.5_7b': 'Qwen1.5-7B', 'yi1.5_6b': 'Yi-1.5-6B', 'bart': 'bart_base',
}
_HF_IDS = {
    "gptj_6b": "EleutherAI/gpt-j-6B", "gptneo_2.7b": "EleutherAI/gpt-neo-2.7B",
    "gpt2_xl": "openai-community/gpt2-xl", "opt_2.7b": "facebook/opt-2.7b",
    "bloom_7b": "bigscience/bloom-7b1", "falcon_7b": "tiiuae/falcon-7b",
    "gemma_7b": "google/gemma-7b", "llama1_13b": "huggyllama/llama-13b",
    "llama2_13b": "TheBloke/Llama-2-13B-fp16", "llama3_8b": "meta-llama/Meta-Llama-3-8B",
    "opt_13b": "facebook/opt-13b", "phi2": "microsoft/phi-2", "mgpt": "ai-forever/mGPT",
    "qwen1.5_7b": "Qwen/Qwen1.5-7B", "yi1.5_6b": "01-ai/Yi-1.5-6B", "bart": "facebook/bart-base",
}


def _resolve_model_path(model_name: str) -> str:
    model_fullname = model_fullnames[model_name]
    local_path = os.path.join("/pretrain_models", model_fullname)
    if os.path.exists(os.path.join(local_path, "config.json")):
        return local_path
    return _HF_IDS.get(model_name, "EleutherAI/gpt-j-6B")


def load_model(model_name):
    model_path = _resolve_model_path(model_name)
    print(f'Loading model {model_name} from {model_path}...')
    model_kwargs = {"device_map": "auto"}
    if model_name in ['gptj_6b', 'llama1_13b', 'llama2_13b', 'llama3_8b', 'falcon_7b', 'bloom_7b', 'opt_13b', 'gemma_7b', 'qwen1.5_7b', 'yi1.5_6b']:
        model_kwargs["torch_dtype"] = torch.float16
    if 'gptj' in model_name:
        model_kwargs["revision"] = 'float16'
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return model


def load_tokenizer(model_name):
    model_path = _resolve_model_path(model_name)
    optional_tok_kwargs = {"padding_side": "right"}
    if "opt-" in model_fullnames.get(model_name, ""):
        optional_tok_kwargs["fast"] = False
    base_tokenizer = AutoTokenizer.from_pretrained(model_path, **optional_tok_kwargs)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_tokenizer


def load_similarity_model(model_name):
    model_path = _resolve_model_path(model_name)
    return bart_score.BARTScorer(device=device, checkpoint=model_path)


def fill_and_mask(text, pct):
    tokens = text.split(' ')
    n_spans = max(0, int(pct * len(tokens)))
    return np.random.choice(range(len(tokens)), size=min(n_spans, len(tokens)), replace=False).tolist()


def apply_extracted_fills(texts, indices_list):
    tokens = [x.split(' ') for x in texts]
    for text, indices in zip(tokens, indices_list):
        for idx in indices:
            if 0 <= idx < len(text):
                text[idx] = ""
    return [" ".join(x) for x in tokens]


def perturb_texts_(texts, pct):
    indices_list = [fill_and_mask(x, pct) for x in texts]
    return apply_extracted_fills(texts, indices_list)


def perturb_texts(texts, pct):
    outputs = []
    for i in range(0, len(texts), 50):
        outputs.extend(perturb_texts_(texts[i : i + 50], pct))
    return outputs


def get_samples(logits, labels, n_samples):
    assert logits.shape[0] == 1 and labels.shape[0] == 1
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    return distrib.sample([n_samples]).permute([1, 2, 0])


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1 and labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    return lprobs.gather(dim=-1, index=labels)


def get_logrank(logits, labels):
    assert logits.shape[0] == 1 and labels.shape[0] == 1
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    if matches.shape[1] != 3:
        return torch.tensor(0.0, device=logits.device)
    ranks, timesteps = matches[:, -1], matches[:, -2]
    if not (timesteps == torch.arange(len(timesteps), device=timesteps.device)).all():
        return torch.tensor(0.0, device=logits.device)
    return -torch.log(ranks.float() + 1).mean()


def get_lastde(log_likelihood, args):
    embed_size = args.embed_size
    epsilon = int(args.epsilon * log_likelihood.shape[1])
    tau_prime = args.tau_prime
    templl = log_likelihood.mean(dim=1)
    aggmde = fastMDE.get_tau_multiscale_DE(ori_data=log_likelihood, embed_size=embed_size, epsilon=epsilon, tau_prime=tau_prime)
    return templl / aggmde


def get_score(logits_ref, logits_score, labels, source_texts, perturbed_texts, base_detection, similarity_model, args):
    assert logits_ref.shape[0] == 1 and logits_score.shape[0] == 1 and labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]
    samples_1 = get_samples(logits_ref, labels, args.n_samples_1)
    log_likelihood_x = get_likelihood(logits_score, labels).mean(dim=1)
    log_rank_x = get_logrank(logits_score, labels).mean().item()
    log_likelihood_x_tilde = get_likelihood(logits_score, samples_1).mean(dim=1)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1).clamp(min=1e-8)
    samples_2 = get_samples(logits_ref, labels, args.n_samples_2)
    log_likelihood_x_temp = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde_temp = get_likelihood(logits_score, samples_2)
    lastde_x = get_lastde(log_likelihood_x_temp, args)
    sampled_lastde = get_lastde(log_likelihood_x_tilde_temp, args)
    miu_tilde_lastde = sampled_lastde.mean()
    sigma_tilde_lastde = sampled_lastde.std().clamp(min=1e-8)
    source_texts_list = [source_texts] * args.copies_number
    values = similarity_model.score(perturbed_texts, source_texts_list, batch_size=args.copies_number)
    mean_values = np.mean(values)
    if base_detection == 'fast_detectgpt':
        output_score = ((log_likelihood_x.squeeze(-1).item() - miu_tilde.item()) / sigma_tilde.item()) * math.exp(-mean_values)
    elif base_detection == 'lastde_doubleplus':
        output_score = ((lastde_x.squeeze(-1).item() - miu_tilde_lastde.item()) / sigma_tilde_lastde.item()) * math.exp(-mean_values)
    elif base_detection == 'lastde':
        output_score = lastde_x.squeeze(-1).item() * math.exp(mean_values)
    elif base_detection == 'lrr':
        output_score = (log_likelihood_x.squeeze(-1).item() / max(log_rank_x, 1e-8)) * math.exp(-mean_values)
    elif base_detection == 'likelihood':
        output_score = log_likelihood_x.squeeze(-1).item() * math.exp(mean_values)
    elif base_detection == 'logrank':
        output_score = log_rank_x * math.exp(mean_values)
    elif base_detection == 'standalone':
        output_score = -mean_values
    else:
        output_score = log_likelihood_x.squeeze(-1).item() * math.exp(mean_values)
    return output_score


def iter_flat_records(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def experiment(args):
    bart_scorer = load_similarity_model(args.similarity_model_name)
    scoring_tokenizer = load_tokenizer(args.scoring_model_name)
    scoring_model = load_model(args.scoring_model_name)
    scoring_model.eval()
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name)
        reference_model = load_model(args.reference_model_name)
        reference_model.eval()
    else:
        reference_tokenizer = reference_model = None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    input_path = args.input
    output_path = args.output
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        n_total = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
    except FileNotFoundError:
        print(f"Input not found: {input_path}")
        return
    total = processed = errors = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for rec in tqdm.tqdm(iter_flat_records(input_path), total=n_total, desc="tocsin", unit="review"):
            total += 1
            text = rec.get("text", "")
            if not text or len(text.strip().split()) < 10:
                out = dict(rec)
                if not getattr(args, "save_text", False):
                    out.pop("text", None)
                md = out.get("detector_metadata") or {}
                key = f"{args.base_detection}_tocsin"
                md[key] = {"detector": "tocsin", "base_detection": args.base_detection, "status": "skipped", "reason": "No text or too short"}
                out["detector_metadata"] = md
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue
            try:
                perturbed_texts = perturb_texts([text] * args.copies_number, args.rho)
                tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False, truncation=True, max_length=2048).to(device)
                labels = tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits_score = scoring_model(**tokenized).logits[:, :-1]
                    if args.reference_model_name == args.scoring_model_name:
                        logits_ref = logits_score
                    else:
                        tokenized_ref = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False, truncation=True, max_length=2048).to(device)
                        logits_ref = reference_model(**tokenized_ref).logits[:, :-1]
                    score_value = get_score(logits_ref, logits_score, labels, text, perturbed_texts, args.base_detection, bart_scorer, args)
                out = dict(rec)
                if not getattr(args, "save_text", False):
                    out.pop("text", None)
                md = out.get("detector_metadata") or {}
                key = f"{args.base_detection}_tocsin"
                md[key] = {"detector": "tocsin", "base_detection": args.base_detection, "reference_model": args.reference_model_name, "scoring_model": args.scoring_model_name, "similarity_model": args.similarity_model_name, "rho": args.rho, "copies_number": args.copies_number, "n_samples_1": args.n_samples_1, "n_samples_2": args.n_samples_2, "embed_size": args.embed_size, "epsilon": args.epsilon, "tau_prime": args.tau_prime, "score": float(score_value)}
                out["detector_metadata"] = md
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                processed += 1
            except Exception as e:
                errors += 1
                out = dict(rec)
                if not getattr(args, "save_text", False):
                    out.pop("text", None)
                md = out.get("detector_metadata") or {}
                md[f"{args.base_detection}_tocsin"] = {"detector": "tocsin", "base_detection": args.base_detection, "status": "error", "error": str(e)}
                out["detector_metadata"] = md
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Done. total={total}, processed={processed}, errors={errors}")
