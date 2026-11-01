"""
Mission 04 — Part 1: Systems Architect (Lead Engineer)
Direct Preference Optimization (DPO) Loss Core & Log-Likelihood Extraction

Squadron Shannon · Station: Systems Architect · Ankit Kumar & Garima Swami
Course: 14013 Conversational AI (BTU Cottbus-Senftenberg, WS 2026/27)
Sector: SEC-04-ORI (Betelgeuse & Orion Molecular Cloud Complex) · AURA Subsystem 04: Plasma Ignition & Ethical Steering Core

DPO (Rafailov et al., NeurIPS 2023) turns the RLHF objective into a classification loss on preference pairs:
    L_DPO = -E[ log sigma( beta * (Delta_w - Delta_l) ) ],   Delta = log pi_theta(y|x) - log pi_ref(y|x)
No reward model and no PPO loop: the policy's own log-ratio is the (implicit) reward.

Everything is pure PyTorch (no TRL). To test the loss on a real model instead of random tensors, this file also
defines a small, fully specified preference world (the "AURA containment protocol") and a tiny GPT that is first
SFT-trained (= the frozen reference pi_ref) and then DPO-trained (= the policy pi_theta). Parts 2 and 3 reuse both.

Run:  python 01_DPO_Loss_Core.py           (all checks + DPO training run, ~1 min on CPU)
      python 01_DPO_Loss_Core.py --quick   (shorter training, for CI / tests)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CKPT = RESULTS / "checkpoints"          # cached tiny models (git-ignored, rebuilt deterministically when missing)


# ----------------------------------------------------------------------------------------------------------
# 1. Token log-probability gather
# ----------------------------------------------------------------------------------------------------------
def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor, label_pad_token_id: int = -100,
                    average_log_prob: bool = False) -> torch.Tensor:
    """Per-sequence log-probability of the label tokens under the logits.

    logits: (B, T, V);  labels: (B, T) with prompt and padding positions set to label_pad_token_id.
    Token t is predicted by position t-1, so logits and labels are shifted by one. Returns shape (B,):
    the sum of log pi(y_t | x, y_<t) over response tokens, or their mean if average_log_prob (used by SimPO).
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and labels {tuple(labels.shape)} do not match")
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    loss_mask = shift_labels != label_pad_token_id
    safe_labels = shift_labels.masked_fill(~loss_mask, 0)           # any valid index; masked out below
    if shift_logits.dtype in (torch.float16, torch.bfloat16):
        shift_logits = shift_logits.float()                             # log-softmax in at least fp32
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
    per_token = per_token * loss_mask
    if average_log_prob:
        return per_token.sum(-1) / loss_mask.sum(-1).clamp(min=1)
    return per_token.sum(-1)


def get_batch_logps_loop(logits: torch.Tensor, labels: torch.Tensor, label_pad_token_id: int = -100,
                         average_log_prob: bool = False) -> torch.Tensor:
    """Slow reference: one Python loop per token. Used only to verify the vectorized gather."""
    out = []
    for b in range(logits.shape[0]):
        total, count = 0.0, 0
        for t in range(1, logits.shape[1]):
            y = int(labels[b, t])
            if y == label_pad_token_id:
                continue
            total += float(torch.log_softmax(logits[b, t - 1].double(), -1)[y])
            count += 1
        out.append(total / max(count, 1) if average_log_prob else total)
    return torch.tensor(out, dtype=torch.float64)


# ----------------------------------------------------------------------------------------------------------
# 2. DPO loss
# ----------------------------------------------------------------------------------------------------------
class DPOLoss(nn.Module):
    """L = -mean log sigma(beta * ((pi_w - ref_w) - (pi_l - ref_l))), computed with logsigmoid (no NaN / inf)."""

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing          # 0 = standard DPO; >0 = conservative DPO (cDPO)

    def forward(self, policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
                reference_chosen_logps: torch.Tensor, reference_rejected_logps: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # The reference policy is frozen: detach so no gradient can ever reach it, even if a caller forgot no_grad.
        reference_chosen_logps = reference_chosen_logps.detach()
        reference_rejected_logps = reference_rejected_logps.detach()
        delta_w = policy_chosen_logps - reference_chosen_logps          # log pi(y_w)/pi_ref(y_w)
        delta_l = policy_rejected_logps - reference_rejected_logps      # log pi(y_l)/pi_ref(y_l)
        logits = self.beta * (delta_w - delta_l)                        # Bradley-Terry logit of the implicit rewards
        eps = self.label_smoothing
        loss = (-(1 - eps) * F.logsigmoid(logits) - eps * F.logsigmoid(-logits)).mean()
        with torch.no_grad():
            r_w, r_l = self.beta * delta_w, self.beta * delta_l
            metrics = {"loss": loss.detach(), "reward_accuracy": (r_w > r_l).float().mean(),
                       "reward_margin": (r_w - r_l).mean(), "chosen_rewards": r_w.mean(),
                       "rejected_rewards": r_l.mean(), "bt_probability": torch.sigmoid(logits).mean()}
        return loss, metrics


def naive_dpo_loss(logits: torch.Tensor) -> torch.Tensor:
    """The textbook form -log(sigmoid(z)): underflows to log(0) = -inf for very negative z. For the stability test."""
    return -torch.log(torch.sigmoid(logits)).mean()


# ----------------------------------------------------------------------------------------------------------
# 3. The AURA containment-protocol preference world (synthetic, fully specified, with a known true reward)
# ----------------------------------------------------------------------------------------------------------
PAD, BOS, SEP, EOS = 0, 1, 2, 3
SENSOR = list(range(4, 20))            # 16 sensor alarm codes (prompt tokens)
ACTION = list(range(20, 64))           # 44 containment action codes (response tokens)
VOCAB = 64
PROMPT_LEN, MIN_RESP, MAX_RESP = 3, 2, 10
SEQ_LEN = 1 + PROMPT_LEN + 1 + MAX_RESP + 1          # BOS s s s SEP y.. EOS = 16


def correct_action(sensor: int) -> int:
    """Each alarm has exactly one correct containment action (3 is coprime with 44, so the map is one-to-one)."""
    return ACTION[(3 * (sensor - SENSOR[0])) % len(ACTION)]


def true_reward(prompt: List[int], response: List[int]) -> float:
    """Ground-truth quality r*(x, y) in [0, 3]: 2 x precision (no junk actions) + 1 x coverage (all alarms handled)."""
    targets = {correct_action(s) for s in prompt}
    if not response:
        return 0.0
    hits = sum(t in targets for t in response)
    return 2.0 * hits / len(response) + len(targets & set(response)) / len(targets)


@dataclass
class PreferenceData:
    prompts: List[List[int]]
    chosen: List[List[int]]
    rejected: List[List[int]]
    p_first: List[float]        # annotator probability that the first sampled response is preferred (for calibration)
    first_won: List[int]        # 1 if the first sampled response became y_w


def sample_response(prompt: List[int], g: torch.Generator) -> List[int]:
    """Behaviour of the unaligned base crew: random length 2-10, each token correct with a per-response skill."""
    length = int(torch.randint(MIN_RESP, MAX_RESP + 1, (1,), generator=g))
    skill = float(torch.rand(1, generator=g))
    targets = [correct_action(s) for s in prompt]
    out = []
    for _ in range(length):
        if float(torch.rand(1, generator=g)) < skill:
            out.append(targets[int(torch.randint(0, len(targets), (1,), generator=g))])
        else:
            out.append(ACTION[int(torch.randint(0, len(ACTION), (1,), generator=g))])
    return out


def sample_prompt(g: torch.Generator) -> List[int]:
    return [SENSOR[i] for i in torch.randperm(len(SENSOR), generator=g)[:PROMPT_LEN].tolist()]


def make_preferences(n: int, seed: int, kappa: float = 3.0, length_bias: float = 0.0) -> PreferenceData:
    """Bradley-Terry annotators: P(y1 > y2) = sigma(kappa (r*(y1) - r*(y2)) + length_bias (|y1| - |y2|)).

    length_bias = 0 -> clean, noisy-but-honest labels.  length_bias > 0 -> annotators who like long answers
    (the verbosity bias that Part 3 diagnoses).
    """
    g = torch.Generator().manual_seed(seed)
    data = PreferenceData([], [], [], [], [])
    while len(data.prompts) < n:
        x = sample_prompt(g)
        y1, y2 = sample_response(x, g), sample_response(x, g)
        if y1 == y2:
            continue
        z = kappa * (true_reward(x, y1) - true_reward(x, y2)) + length_bias * (len(y1) - len(y2))
        p1 = 1.0 / (1.0 + math.exp(-z))
        first = int(float(torch.rand(1, generator=g)) < p1)
        data.prompts.append(x)
        data.chosen.append(y1 if first else y2)
        data.rejected.append(y2 if first else y1)
        data.p_first.append(p1)
        data.first_won.append(first)
    return data


def encode(prompts: List[List[int]], responses: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Token ids (B, SEQ_LEN) and labels with the prompt (BOS..SEP) and padding set to -100."""
    ids = torch.full((len(prompts), SEQ_LEN), PAD, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for i, (x, y) in enumerate(zip(prompts, responses)):
        seq = [BOS] + x + [SEP] + y + [EOS]
        ids[i, :len(seq)] = torch.tensor(seq)
        labels[i, 1 + PROMPT_LEN + 1:len(seq)] = torch.tensor(y + [EOS])
    return ids, labels


# ----------------------------------------------------------------------------------------------------------
# 4. A tiny GPT (pre-LN decoder, causal attention) — the language model that gets aligned
# ----------------------------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int = VOCAB, d: int = 128, layers: int = 2, heads: int = 4, max_len: int = SEQ_LEN):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n = ids.shape[1]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=ids.device), 1)
        x = self.tok(ids) + self.pos(torch.arange(n, device=ids.device))
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln(x))


def sft_reference(seed: int = 0, steps: int = 1500, quick: bool = False) -> TinyGPT:
    """pi_ref: supervised fine-tuning on the base crew's raw (unranked) responses. Cached on disk."""
    steps = 300 if quick else steps
    path = CKPT / f"ref_seed{seed}_steps{steps}.pt"
    torch.manual_seed(seed)
    model = TinyGPT()
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
        return freeze(model)
    g = torch.Generator().manual_seed(seed + 1000)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    for _ in range(steps):
        xs = [sample_prompt(g) for _ in range(128)]
        ids, labels = encode(xs, [sample_response(x, g) for x in xs])
        loss = -get_batch_logps(model(ids), labels, average_log_prob=True).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return freeze(model)


def freeze(model: nn.Module) -> nn.Module:
    """Turn a model into a frozen reference: eval mode, no gradients, no stale .grad buffers."""
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    return model.eval()


def trainable_copy(model: nn.Module) -> nn.Module:
    policy = copy.deepcopy(model).train()
    for p in policy.parameters():
        p.requires_grad_(True)
    return policy


def pair_logps(model: nn.Module, prompts, chosen, rejected, average: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """One forward pass over [chosen; rejected] (the standard DPO concatenation trick)."""
    ids_w, lab_w = encode(prompts, chosen)
    ids_l, lab_l = encode(prompts, rejected)
    logps = get_batch_logps(model(torch.cat([ids_w, ids_l])), torch.cat([lab_w, lab_l]), average_log_prob=average)
    return logps[:len(prompts)], logps[len(prompts):]


def train_dpo(ref: TinyGPT, data: PreferenceData, beta: float = 0.1, epochs: int = 4, batch: int = 64,
              lr: float = 5e-4, seed: int = 0, log_every: int = 10) -> Tuple[TinyGPT, List[Dict[str, float]]]:
    """Standard DPO: policy initialised from pi_ref, pi_ref frozen. Returns the policy and a training log."""
    torch.manual_seed(seed)
    policy = trainable_copy(ref)
    freeze(ref)
    criterion = DPOLoss(beta)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    n, log, step = len(data.prompts), [], 0
    for _ in range(epochs):
        for idx in torch.randperm(n).split(batch):
            sel = idx.tolist()
            xs = [data.prompts[i] for i in sel]
            yw = [data.chosen[i] for i in sel]
            yl = [data.rejected[i] for i in sel]
            pw, pl = pair_logps(policy, xs, yw, yl)
            with torch.no_grad():
                rw, rl = pair_logps(ref, xs, yw, yl)
            loss, m = criterion(pw, pl, rw, rl)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % log_every == 0:
                log.append({"step": step, **{k: float(v) for k, v in m.items()}})
            step += 1
    return policy.eval(), log


@torch.no_grad()
def generate(model: TinyGPT, prompts: List[List[int]], seed: int = 0, temperature: float = 1.0) -> List[List[int]]:
    """Sample responses token by token until EOS (or the length limit)."""
    g = torch.Generator().manual_seed(seed)
    outs = []
    for x in prompts:
        seq = [BOS] + x + [SEP]
        for _ in range(MAX_RESP + 1):
            logits = model(torch.tensor([seq]))[0, -1] / temperature
            logits[[PAD, BOS, SEP] + SENSOR] = -float("inf")          # only actions or EOS are valid here
            if len(seq) - (PROMPT_LEN + 2) < MIN_RESP:
                logits[EOS] = -float("inf")
            nxt = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=g))
            if nxt == EOS:
                break
            seq.append(nxt)
        outs.append(seq[PROMPT_LEN + 2:])
    return outs


def sample_quality(model: TinyGPT, n: int = 300, seed: int = 7) -> Dict[str, float]:
    g = torch.Generator().manual_seed(seed)
    prompts = [sample_prompt(g) for _ in range(n)]
    outs = generate(model, prompts, seed=seed)
    return {"mean_true_reward": sum(true_reward(x, y) for x, y in zip(prompts, outs)) / n,
            "mean_length": sum(len(y) for y in outs) / n}


# ----------------------------------------------------------------------------------------------------------
# 5. Verification
# ----------------------------------------------------------------------------------------------------------
def check_gather() -> Dict[str, float]:
    torch.manual_seed(42)
    logits = torch.randn(4, 16, 128, dtype=torch.float64)
    labels = torch.randint(0, 128, (4, 16))
    labels[:, :4] = -100
    labels[1, 12:] = -100                                          # trailing padding on one row
    fast, slow = get_batch_logps(logits, labels), get_batch_logps_loop(logits, labels)
    fast_avg, slow_avg = (get_batch_logps(logits, labels, average_log_prob=True),
                          get_batch_logps_loop(logits, labels, average_log_prob=True))
    padded_logits = torch.cat([logits, torch.randn(4, 5, 128, dtype=torch.float64)], 1)
    padded_labels = torch.cat([labels, torch.full((4, 5), -100)], 1)
    return {"max_abs_diff_sum_vs_loop": float((fast - slow).abs().max()),
            "max_abs_diff_avg_vs_loop": float((fast_avg - slow_avg).abs().max()),
            "max_abs_diff_extra_padding": float((get_batch_logps(padded_logits, padded_labels) - fast).abs().max())}


def check_stability() -> List[Dict[str, float]]:
    rows = []
    for z in [0.0, 20.0, -20.0, -100.0, -200.0, -1000.0]:
        t = torch.tensor([z], dtype=torch.float32)
        rows.append({"logit": z, "naive_-log(sigmoid)": float(naive_dpo_loss(t)), "logsigmoid": float(-F.logsigmoid(t).mean()),
                     "exact": max(-z, 0.0) + math.log1p(math.exp(-abs(z)))})
    return rows


def check_gradient_formula(beta: float = 0.1) -> Dict[str, float]:
    """dL/dDelta_w = -beta * sigma(-z) / B and dL/dDelta_l = +beta * sigma(-z) / B with z = beta (Delta_w - Delta_l)."""
    torch.manual_seed(1)
    pw = torch.randn(8, dtype=torch.float64, requires_grad=True)
    pl = torch.randn(8, dtype=torch.float64, requires_grad=True)
    rw, rl = torch.randn(8, dtype=torch.float64), torch.randn(8, dtype=torch.float64)
    loss, _ = DPOLoss(beta)(pw, pl, rw, rl)
    loss.backward()
    z = beta * ((pw - rw) - (pl - rl)).detach()
    analytic = -beta * torch.sigmoid(-z) / 8
    return {"max_abs_diff_grad_chosen": float((pw.grad - analytic).abs().max()),
            "max_abs_diff_grad_rejected": float((pl.grad + analytic).abs().max())}


def check_frozen_reference(ref: TinyGPT, data: PreferenceData) -> Dict[str, float]:
    """Backprop through a real policy/reference pair: only the policy may receive gradients or change."""
    policy = trainable_copy(ref)
    freeze(ref)
    ref_before = {k: v.clone() for k, v in ref.state_dict().items()}
    xs, yw, yl = data.prompts[:32], data.chosen[:32], data.rejected[:32]
    pw, pl = pair_logps(policy, xs, yw, yl)
    rw, rl = pair_logps(ref, xs, yw, yl)                         # deliberately without no_grad
    loss, m = DPOLoss(0.1)(pw, pl, rw, rl)
    loss.backward()
    policy_grads = [p.grad for p in policy.parameters()]
    torch.optim.SGD(policy.parameters(), lr=0.1).step()
    return {"loss_at_init": float(loss.detach()), "log_2": math.log(2), "margin_at_init": float(m["reward_margin"]),
            "policy_params_with_grad": sum(g is not None for g in policy_grads),
            "policy_params_total": len(policy_grads),
            "policy_grad_norm": float(torch.sqrt(sum((g ** 2).sum() for g in policy_grads if g is not None))),
            "ref_params_with_grad": sum(p.grad is not None for p in ref.parameters()),
            "ref_max_abs_change_after_step": max(float((ref.state_dict()[k] - v).abs().max()) for k, v in ref_before.items()),
            "policy_max_abs_change_after_step": max(float((policy.state_dict()[k] - ref_before[k]).abs().max()) for k in ref_before)}


@torch.no_grad()
def heldout_accuracy(policy: TinyGPT, ref: TinyGPT, data: PreferenceData, beta: float = 0.1) -> Dict[str, float]:
    pw, pl = pair_logps(policy, data.prompts, data.chosen, data.rejected)
    rw, rl = pair_logps(ref, data.prompts, data.chosen, data.rejected)
    margin = beta * ((pw - rw) - (pl - rl))
    truth = torch.tensor([true_reward(x, w) - true_reward(x, l) for x, w, l in zip(data.prompts, data.chosen, data.rejected)])
    keep = truth != 0
    # a tie (margin exactly 0, e.g. policy == reference) counts as a coin flip: half credit
    hit = (margin > 0).float() + 0.5 * (margin == 0).float()
    agree_truth = torch.where(truth > 0, hit, 1 - hit)[keep].mean()
    return {"accuracy_vs_annotator_labels": float(hit.mean()),
            "accuracy_vs_true_reward": float(agree_truth), "mean_margin": float(margin.mean())}


def plot_training(log: List[Dict[str, float]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [r["step"] for r in log]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(steps, [r["loss"] for r in log], color="#0369a1"); ax[0].axhline(math.log(2), ls="--", c="grey", lw=1)
    ax[0].set_title("DPO loss (dashed: log 2 at init)")
    ax[1].plot(steps, [r["reward_accuracy"] for r in log], color="#047857"); ax[1].set_title("batch reward accuracy")
    ax[2].plot(steps, [r["chosen_rewards"] for r in log], label="chosen r(x, y_w)", color="#047857")
    ax[2].plot(steps, [r["rejected_rewards"] for r in log], label="rejected r(x, y_l)", color="#b91c1c")
    ax[2].set_title("implicit rewards beta * log ratio"); ax[2].legend(frameon=False)
    for a in ax:
        a.set_xlabel("step"); a.grid(alpha=.3)
    fig.tight_layout(); FIGURES.mkdir(exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def main(quick: bool = False) -> Dict:
    print("=" * 100)
    print("AURA Subsystem 04 · Part 1 · DPO Loss Core & Log-Likelihood Extraction (Squadron Shannon)")
    print("=" * 100)
    g = check_gather()
    print(f"[1] get_batch_logps (vectorized gather) vs per-token Python loop: max |diff| sum = {g['max_abs_diff_sum_vs_loop']:.2e}, "
          f"mean = {g['max_abs_diff_avg_vs_loop']:.2e}; 5 extra pad columns change it by {g['max_abs_diff_extra_padding']:.2e}")
    stab = check_stability()
    print("[2] Numerical stability of -log sigma(z):")
    print(f"    {'z':>8} | {'naive -log(sigmoid(z))':>22} | {'-logsigmoid(z)':>14} | {'exact':>10}")
    for r in stab:
        print(f"    {r['logit']:>8.0f} | {r['naive_-log(sigmoid)']:>22.4f} | {r['logsigmoid']:>14.4f} | {r['exact']:>10.4f}")
    gr = check_gradient_formula()
    print(f"[3] autograd vs analytic dL/dDelta = -/+ beta sigma(-z)/B: max |diff| = "
          f"{max(gr['max_abs_diff_grad_chosen'], gr['max_abs_diff_grad_rejected']):.2e}")

    ref = sft_reference(quick=quick)
    train = make_preferences(1000 if quick else 4000, seed=11)
    test = make_preferences(500 if quick else 2000, seed=12)
    fr = check_frozen_reference(ref, train)
    print(f"[4] Real TinyGPT policy/reference: loss at init = {fr['loss_at_init']:.4f} (log 2 = {fr['log_2']:.4f}), margin {fr['margin_at_init']:+.1e}")
    print(f"    policy tensors with grad {fr['policy_params_with_grad']}/{fr['policy_params_total']} (grad norm {fr['policy_grad_norm']:.3f}); "
          f"reference tensors with grad {fr['ref_params_with_grad']}; after one optimizer step the reference moved "
          f"{fr['ref_max_abs_change_after_step']:.1e}, the policy {fr['policy_max_abs_change_after_step']:.1e}")

    before = {**heldout_accuracy(ref, ref, test), **sample_quality(ref)}
    policy, log = train_dpo(ref, train, beta=0.1, epochs=2 if quick else 4)
    after = {**heldout_accuracy(policy, ref, test), **sample_quality(policy)}
    print(f"\n[5] DPO training on {len(train.prompts)} AURA preference pairs (beta 0.1, AdamW 5e-4, {log[-1]['step'] + 1} steps), "
          f"evaluated on {len(test.prompts)} held-out pairs:")
    print(f"    {'':<26} | {'acc vs labels':>13} | {'acc vs true r*':>14} | {'sampled r* (0-3)':>16} | {'sampled length':>14}")
    for name, r in [("pi_ref (SFT, before DPO)", before), ("pi_theta (after DPO)", after)]:
        print(f"    {name:<26} | {r['accuracy_vs_annotator_labels']:>13.1%} | {r['accuracy_vs_true_reward']:>14.1%} | "
              f"{r['mean_true_reward']:>16.3f} | {r['mean_length']:>14.2f}")
    ok = (g["max_abs_diff_sum_vs_loop"] < 1e-9 and g["max_abs_diff_extra_padding"] < 1e-12
          and all(math.isfinite(r["logsigmoid"]) for r in stab) and max(gr.values()) < 1e-12
          and abs(fr["loss_at_init"] - math.log(2)) < 1e-5 and fr["ref_params_with_grad"] == 0
          and fr["ref_max_abs_change_after_step"] == 0 and fr["policy_params_with_grad"] == fr["policy_params_total"]
          and after["accuracy_vs_true_reward"] > before["accuracy_vs_true_reward"])
    print(f"\n{'[PASS]' if ok else '[FAIL]'} DPO core: exact gather, stable loss, analytic gradient, frozen reference, policy learns the preferences")
    suffix = "_quick" if quick else ""
    plot_training(log, FIGURES / f"01_dpo_training{suffix}.png")
    result = {"gather": g, "stability": stab, "gradient": gr, "frozen_reference": fr, "before_dpo": before,
              "after_dpo": after, "training_log": log, "n_train": len(train.prompts), "n_test": len(test.prompts),
              "meta": {"torch": torch.__version__, "machine": platform.machine(), "python": platform.python_version()},
              "all_passed": bool(ok)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"01_dpo_core{suffix}.json").write_text(json.dumps(result, indent=2))
    print(f"Results -> results/01_dpo_core{suffix}.json · figure -> figures/01_dpo_training{suffix}.png")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
