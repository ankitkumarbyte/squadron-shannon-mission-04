"""Unit tests for Mission 04 (run from the repo root:  python -m pytest -q tests/)."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

PS = Path(__file__).resolve().parents[1] / "docs" / "weeks" / "week-04" / "problem_set"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, PS / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


core = _load("dpo_core", "01_DPO_Loss_Core.py")
rew = _load("implicit_reward", "02_Implicit_Reward_Evaluator.py")
lb = _load("length_bias", "03_Length_Bias_and_SimPO.py")
st = _load("steering", "04_Activation_Steering_and_Audit.py")


# ---------------------------------------------------------------- Part 1
def test_gather_matches_loop_and_ignores_padding():
    g = core.check_gather()
    assert g["max_abs_diff_sum_vs_loop"] < 1e-10
    assert g["max_abs_diff_avg_vs_loop"] < 1e-10
    assert g["max_abs_diff_extra_padding"] < 1e-10


def test_gather_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        core.get_batch_logps(torch.zeros(2, 5, 7), torch.zeros(2, 4, dtype=torch.long))


def test_dpo_loss_is_log2_when_policy_equals_reference():
    x = torch.randn(6)
    loss, m = core.DPOLoss(0.1)(x, x - 1, x, x - 1)
    assert abs(float(loss) - math.log(2)) < 1e-6 and float(m["reward_margin"]) == 0


def test_dpo_loss_stable_for_extreme_logits():
    for z in (-1000.0, -200.0, 200.0):
        t = torch.tensor([z])
        assert math.isinf(float(core.naive_dpo_loss(t))) or z > 0
        assert math.isfinite(float(-F.logsigmoid(t).mean()))
    loss, _ = core.DPOLoss(1.0)(torch.tensor([-5000.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    assert abs(float(loss) - 5000.0) < 1e-3


def test_dpo_gradient_matches_closed_form():
    g = core.check_gradient_formula()
    assert max(g.values()) < 1e-12


def test_reference_stays_frozen():
    ref = core.sft_reference(quick=True)
    fr = core.check_frozen_reference(ref, core.make_preferences(64, seed=3))
    assert fr["ref_params_with_grad"] == 0 and fr["ref_max_abs_change_after_step"] == 0
    assert fr["policy_params_with_grad"] == fr["policy_params_total"] and fr["policy_max_abs_change_after_step"] > 0
    assert abs(fr["loss_at_init"] - math.log(2)) < 1e-5


def test_true_reward_and_encoding():
    x = [4, 5, 6]
    perfect = [core.correct_action(s) for s in x]
    assert core.true_reward(x, perfect) == 3.0
    assert core.true_reward(x, [core.ACTION[-1]] * 3) < 1.0 or core.correct_action(4) == core.ACTION[-1]
    ids, labels = core.encode([x], [perfect])
    assert (labels[0, :5] == -100).all() and labels[0, 5:9].tolist() == perfect + [core.EOS]


def test_dpo_training_improves_true_quality():
    ref = core.sft_reference(quick=True)
    data = core.make_preferences(600, seed=21)
    test = core.make_preferences(300, seed=22)
    policy, log = core.train_dpo(ref, data, epochs=2)
    acc = core.heldout_accuracy(policy, ref, test)
    assert acc["accuracy_vs_true_reward"] > 0.7
    assert log[-1]["loss"] < math.log(2)


# ---------------------------------------------------------------- Part 2
def test_calibration_curve_and_metrics():
    probs = torch.tensor([0.1, 0.1, 0.9, 0.9])
    outcomes = torch.tensor([0.0, 0.0, 1.0, 1.0])
    m = rew.calibration_metrics(probs, outcomes)
    assert abs(m["ece"] - 0.1) < 1e-6 and m["accuracy"] == 1.0
    curve = rew.calibration_curve([0.05, 0.95], [0, 1], num_bins=10)
    assert curve[0]["count"] == 1 and curve[-1]["count"] == 1


def test_temperature_recovers_known_scale():
    torch.manual_seed(0)
    true_logit = torch.randn(4000) * 2
    outcomes = torch.bernoulli(torch.sigmoid(true_logit))
    t = rew.fit_temperature(true_logit * 3, outcomes)          # model 3x over-confident
    assert 2.5 < t < 3.5


def test_implicit_rewards_match_dpo_definition():
    ref = core.sft_reference(quick=True)
    data = core.make_preferences(32, seed=5)
    policy, _ = core.train_dpo(ref, core.make_preferences(200, seed=6), epochs=1)
    batch = {"prompts": data.prompts, "chosen": data.chosen, "rejected": data.rejected}
    ev = rew.evaluate_implicit_rewards(policy, ref, batch, beta=0.1)
    pw, pl = core.pair_logps(policy, data.prompts, data.chosen, data.rejected)
    rw, rl = core.pair_logps(ref, data.prompts, data.chosen, data.rejected)
    assert torch.allclose(ev["reward_w"], 0.1 * (pw - rw).detach(), atol=1e-6)
    assert torch.allclose(ev["p_win"], torch.sigmoid(ev["reward_w"] - ev["reward_l"]))


# ---------------------------------------------------------------- Part 3
def test_pearson():
    assert abs(lb.pearson([1, 2, 3], [2, 4, 6]) - 1) < 1e-12
    assert abs(lb.pearson([1, 2, 3], [3, 2, 1]) + 1) < 1e-12
    assert lb.pearson([1, 1, 1], [1, 2, 3]) == 0.0


def test_simpo_loss_formula():
    w, l = torch.tensor([-0.15, -0.22]), torch.tensor([-0.45, -0.38])
    loss, m = lb.SimPOLoss(2.0, 0.5)(w, l)
    assert abs(float(loss) - float(-F.logsigmoid(2 * (w - l) - 0.5).mean())) < 1e-7
    assert float(m["margin_pass_rate"]) == 0.5          # 2(0.30) - 0.5 > 0 but 2(0.16) - 0.5 < 0


def test_rdpo_length_term_shifts_logit():
    zero = torch.zeros(1)
    loss_long, _ = lb.LengthRegularizedDPOLoss(0.1, 0.35)(zero, zero, zero, zero, torch.tensor([9]), torch.tensor([3]))
    assert abs(float(loss_long) - float(-F.logsigmoid(torch.tensor(0.35 * 6)))) < 1e-6


def test_verbose_annotators_prefer_long_answers():
    clean = lb.dataset_diagnostics(core.make_preferences(1500, seed=11))
    verbose = lb.dataset_diagnostics(core.make_preferences(1500, seed=11, length_bias=0.35))
    assert verbose["chosen_is_longer"] > clean["chosen_is_longer"] + 0.05


def test_analytic_vram_reference_is_extra_weights():
    a = lb.analytic_vram()
    assert abs((a["dpo_gb"] - a["simpo_gb"]) - a["weights_gb"]) < 1e-9


# ---------------------------------------------------------------- Part 4 (no download needed)
class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, hidden_states):
        return self.layer(hidden_states)


def test_steering_hook_adds_vector_and_is_removable():
    torch.manual_seed(0)
    toy = _Toy()
    x = torch.randn(2, 3, 4)
    v = torch.tensor([1.0, -2.0, 0.5, 0.0])
    base = toy(x)
    hook = st.ActivationSteeringHook(v, alpha=1.5).register(toy.layer)
    steered = toy(x)
    hook.remove()
    assert torch.allclose(steered, toy.layer(x + 1.5 * v))
    assert torch.allclose(toy(x), base)
    with st.ActivationSteeringHook(v, 2.0).register(toy.layer):
        assert not torch.allclose(toy(x), base)
    assert torch.allclose(toy(x), base)


def test_steering_hook_keyword_hidden_states():
    toy = _Toy()
    x = torch.randn(1, 2, 4)
    v = torch.ones(4)
    with st.ActivationSteeringHook(v, 1.0).register(toy):
        out = toy(hidden_states=x)
    assert torch.allclose(out, toy.layer(x + 1.0))


def test_sycophancy_items_are_balanced():
    items = st.sycophancy_items(st.FACTS[:5])
    assert len(items) == 20
    assert sum(it["honest_yes"] for it in items) == 10
    pairs = st.contrast_pairs(st.FACTS[:3])
    assert sum(h.startswith("Yes") for _, h, _ in pairs) == sum(h.startswith("No") for _, h, _ in pairs)


def test_math_questions_have_correct_answers():
    for q, a in st.math_questions(30):
        nums = [int(t) for t in q.replace("?", " ").split() if t.isdigit()]
        if "+" in q:
            assert a == nums[0] + nums[1]
        elif "-" in q:
            assert a == nums[0] - nums[1] and a >= 0
        else:
            assert a == nums[0] * nums[1]


def _qwen_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(st.MODEL_ID, "model.safetensors"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _qwen_cached(), reason="Qwen2.5-0.5B-Instruct not in the local Hugging Face cache")
def test_real_model_hook_hygiene():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(st.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(st.MODEL_ID, dtype=torch.float32).eval()
    user = st.PRESSURE.format(claim=st.FACTS[0][1]) + st.ASK["right"]
    before = st.p_yes(model, tok, [user])
    vec = st.extract_steering_vector(model, tok, st.contrast_pairs(st.FACTS[:2]), 8)
    assert vec.shape == (model.config.hidden_size,)
    with st.ActivationSteeringHook(vec, 2.0).register(st.decoder_layers(model)[8]):
        steered = st.p_yes(model, tok, [user])
    after = st.p_yes(model, tok, [user])
    assert float((after - before).abs()) < 1e-6 and float((steered - before).abs()) > 1e-3
