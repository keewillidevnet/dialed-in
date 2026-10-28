"""Tests for C4: structural levers, window check, text controls, perplexity, outcomes, runner and analysis."""
import hashlib
import json
import types

import numpy as np
import torch
from torch import nn

from dialedin import metrics, metrics_c4, analyze_c4, run_c4
from dialedin.hooks import HeadProbe
from dialedin.wave import WaveInjector

T, T1 = metrics.WINDOW_END, metrics.WINDOW_START
D, HEADS, VOCAB = 64, 4, 1001


class Attn(nn.Module):
    def __init__(self, local):
        super().__init__()
        self.is_local_attention = local
        self.sliding_window = 2048 if local else None
        self.o_proj = nn.Linear(D, D, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(D) * 0.01)

    def forward(self, h):
        w = self.sliding_window or 64
        c = torch.cumsum(h.double(), dim=1)                # float64, then cast: out-of-window tokens cancel exactly
        sh = torch.zeros_like(c)
        sh[:, w:] = c[:, :-w]
        return self.o_proj(((c - sh) / w).to(h.dtype))


class Layer(nn.Module):
    def __init__(self, local):
        super().__init__()
        self.self_attn = Attn(local)
        self.post_attention_layernorm = nn.Identity()

    def forward(self, h):
        return h + self.post_attention_layernorm(self.self_attn(h))


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(VOCAB, D)
        self.layers = nn.ModuleList([Layer(i != 3) for i in range(6)])
        self.config = types.SimpleNamespace(sliding_window=2048, rope_parameters={"rope_theta": 500000.0},
                                            output_multiplier=0.5, final_logit_softcapping=3.0)
        self.rotary_emb = nn.Module()
        self.rotary_emb.register_buffer("inv_freq", 1.0 / (500000.0 ** (torch.arange(0, 16, 2).float() / 16)))
        self.head = nn.Linear(D, VOCAB, bias=False)

    def forward(self, input_ids, use_cache=False):
        pos = torch.arange(input_ids.shape[1], dtype=torch.float)
        rot = torch.cos(pos[:, None] * self.rotary_emb.inv_freq[None, :])          # positions enter through inv_freq
        h = self.embed(input_ids) + 0.01 * nn.functional.pad(rot, (0, D - rot.shape[1]))
        for l in self.layers:
            h = l(h)
        return types.SimpleNamespace(last_hidden_state=h)


class ToyFull:
    """Stands in for the full model: parameters, the output head, and logits from the toy's own forward."""
    def __init__(self, lm):
        self.lm = lm
        self.parameters = lm.parameters
        self.get_output_embeddings = lambda: lm.head
        self.config = types.SimpleNamespace(text_config=lm.config)

    def __call__(self, input_ids, use_cache=False):         # written out independently of run_c4.Session.logits
        c = self.lm.config
        x = self.lm.head(self.lm(input_ids).last_hidden_state) * c.output_multiplier
        return types.SimpleNamespace(logits=c.final_logit_softcapping * torch.tanh(x / c.final_logit_softcapping))


def toy_session(tmp_path, periods=(3, 12, 48, 192, 768)):     # multiples of 3: the toy averages never null them
    lm = ToyLM()
    s = object.__new__(run_c4.Session)
    h = lambda t: hashlib.sha256(t.encode()).hexdigest()
    rng = np.random.default_rng(0)
    toks = {g: rng.integers(1, 900, size=T + 10).tolist() for g in (1342, 7)}
    s.texts = {1342: "x", 7: "y"}
    s.tok = lambda text, add_special_tokens=False: {"input_ids": toks[{"x": 1342, "y": 7}[text]]}
    s.spec = {"inputs": {"prompt_sets": {"A": {7: "y"}}, "text_sha256": {1342: h("x"), 7: h("y")}},
              "drive": {"periods_N": list(periods)}, "length_tokens": T,
              "budget": {"max_gpu_hours": 1, "seconds_per_pass_estimate": 0.01}}
    s.spec["inputs"]["length_tokens"] = T
    s.model = ToyFull(lm)
    s.decoder, s.layers = lm, lm.layers
    s.platform = {"global_layers": [3]}
    s.ledger = run_c4.Ledger(str(tmp_path / "ledger"), "testseal0000", 3600)
    s.seal_hash, s.code = "test", {"f": "h"}
    run_c4.Session._setup(s)
    for hd in s.probe.handles:
        hd.remove()
    s.probe = HeadProbe(lm.layers, s.layer_ids, HEADS, s.t1)
    d = torch.zeros(D); d[0] = 1.0
    s.injector.remove(); s.direction = d
    s.injector = WaveInjector(lm.layers[1], d)
    return s


def test_text_ids():
    ids = np.array([5, 3, 5, 9, 5, 3])
    sh = run_c4.text_ids(ids, "shuffled")
    assert sorted(sh) == sorted(ids) and not np.array_equal(sh, ids)
    assert (run_c4.text_ids(ids, "constant") == 5).all()
    assert run_c4.text_ids(ids, "book") is ids


def test_set_structure_and_exact_restore(tmp_path):
    s = toy_session(tmp_path)
    inv0 = s.rotary.inv_freq.clone()
    s.set_condition("W4096")
    assert s.decoder.config.sliding_window == 4096
    assert all(s.layers[i].self_attn.sliding_window == 4096 for i in s.local_ids)
    s.set_condition("R2.0")
    assert s.decoder.config.sliding_window == 2048
    assert torch.allclose(s.rotary.inv_freq, 1.0 / (1e6 ** (torch.arange(0, 16, 2).float() / 16)))
    s.set_condition("BASE")
    assert torch.equal(s.rotary.inv_freq, inv0)


def test_window_check(tmp_path):
    s = toy_session(tmp_path)
    for w in (1024, 2048, 4096):
        s.set_structure(w, 1.0)
        assert s.window_check(w)["as_expected"], w
    s.set_structure(2048, 1.0)
    s.layers[0].self_attn.sliding_window = 1024                  # config says 2048 but the layer ignores it
    assert not s.window_check(2048)["as_expected"]


def test_setup_checks(tmp_path):
    s = toy_session(tmp_path)
    assert all(s.rope_check(f)["as_expected"] for f in (0.5, 2.0))
    assert s.logits_check()["as_expected"]
    assert s.softcap == 3.0 and s.multiplier == 0.5
    s.multiplier = None                                           # a skipped multiplier must fail the check
    assert not s.logits_check()["as_expected"]
    s.multiplier, s.softcap = 0.5, 30.0                           # so must a wrong softcap
    assert not s.logits_check()["as_expected"]


def test_missing_softcap_stops_setup(tmp_path, monkeypatch):
    orig = ToyLM.__init__

    def init(self):
        orig(self)
        del self.config.final_logit_softcapping
    monkeypatch.setattr(ToyLM, "__init__", init)
    try:
        toy_session(tmp_path)
        assert False, "a missing softcap must stop setup"
    except SystemExit as e:
        assert "softcap" in str(e)


def test_rope_base_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(run_c4, "ROPE_THETA", 10000.0)
    try:
        toy_session(tmp_path)
        assert False, "a wrong RoPE base must stop setup"
    except SystemExit as e:
        assert "RoPE base" in str(e)


def test_perplexity_is_finite_and_matches_direct(tmp_path):
    s = toy_session(tmp_path)
    ids, _ = s.ids(1342)
    m, ppl = s.baseline(ids)
    with torch.no_grad():
        h = s.decoder(ids).last_hidden_state[0]
        lp = s.logits(h[run_c4.PPL_FROM:T - 1]).log_softmax(-1)
        direct = float(np.exp(-lp.gather(1, ids[0, run_c4.PPL_FROM + 1:, None]).mean()))
    assert np.isfinite(ppl) and abs(ppl / direct - 1) < 1e-4 and m > 0


def test_lever_outcomes():
    t = {"W1024": 0.10, "BASE": 0.14, "W4096": 0.18}
    o = metrics_c4.lever_outcome(t, [0.08] * 5, 1.02, 1.05, "window")
    assert o["outcome"] == "D1" and o["order"] == "+"
    assert metrics_c4.lever_outcome(t, [0.08] * 5, 1.02, 1.30, "window")["outcome"] == "D2"
    assert metrics_c4.lever_outcome(t, [0.08, 0.08, 0.08, -0.01, -0.01], 1.0, 1.0, "window")["outcome"] == "D0"
    assert metrics_c4.lever_outcome({"W1024": 0.135, "BASE": 0.14, "W4096": 0.145}, [0.01] * 5, 1, 1, "window")["outcome"] == "D0"
    assert metrics_c4.lever_outcome({"W1024": None, "BASE": 0.14, "W4096": 0.145}, [], 1, 1, "window")["outcome"] == "U"


def test_text_outcomes():
    assert metrics_c4.text_outcome(0.14, 0.12, 0.09)["outcome"] == "A"
    assert metrics_c4.text_outcome(0.14, 0.12, 0.02)["outcome"] == "X"
    assert metrics_c4.text_outcome(0.14, 0.12, -0.05)["outcome"] == "X"
    assert metrics_c4.text_outcome(0.14, 0.06, 0.12)["outcome"] == "M"
    assert metrics_c4.text_outcome(0.14, None, 0.12)["outcome"] == "U"


def test_phases_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(run_c4, "PHASE_A_N", (3, 768))
    s = toy_session(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    with torch.no_grad():
        run_c4.phase_a(s, str(out))
    a = json.load(open(out / "phaseA.json"))
    assert a["stopped"] is None and all(c["as_expected"] for c in a["window_checks"] + a["rope_checks"])
    assert a["logits_check"]["as_expected"]
    assert len(a["rows"]) == 7
    with torch.no_grad():
        run_c4.phase_b(s, str(out))
    book = json.load(open(out / "book_A_7.json"))
    assert sorted(book["per_condition"]) == sorted(analyze_c4.CONDITIONS)
    assert all(v == 0 for v in book["sham_differences"].values()) and book["sham_differences_close"] == 0
    assert book["token_ids_sha256"]["SHUF"] != book["token_ids_sha256"]["BASE"]
    assert s.decoder.config.sliding_window == 2048                  # restored after the book
    spec = {"drive": {"periods_N": s.grid}, "inputs": s.spec["inputs"]}
    r = analyze_c4.analyze(str(out), spec, "test", {"f": "h"})
    assert r["complete"] and set(r["C4"]) == {"window", "rope", "text"}
    w = r["C4"]["window"]
    assert w["outcome"] in ("D1", "D2", "D0") and w["tilts"]["high"] != w["tilts"]["low"]   # the toy's window moves gain
    json.dumps(r)


def test_failed_setup_check_stops_by_rule(tmp_path, monkeypatch):
    s = toy_session(tmp_path)
    monkeypatch.setattr(s, "logits_check", lambda: {"max_abs_diff": 1.0, "max_abs_logit": 1.0, "tolerance": 0.005,
                                                    "as_expected": False})
    out = tmp_path / "run"; out.mkdir()
    try:
        with torch.no_grad():
            run_c4.phase_a(s, str(out))
        assert False, "a failed setup check must stop Phase A"
    except SystemExit as e:
        assert "perplexity path" in str(e)
    a = json.load(open(out / "phaseA.json"))
    assert a["stopped"].startswith("setup checks failed") and a["rows"] == []
    assert not list(out.glob("phaseA.FAILED.*"))
