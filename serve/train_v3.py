"""REINFORCE 트레이너 v3 — 강제 복잡도 환경 (확률적 업체 가용성).

목표: 상황민감·고복잡 정책. 매일 각 업체가 확률적으로 품절(unavailable)됨.
가용성을 상태에 노출 + 액션 마스킹 → 정책이 "가용 업체 중 최적" 컨틴전시 플랜 학습.
수요예측 feature(긴급도)와 결합 → 재고/수요/가용성에 따라 다른 업체 선택.

원본 env 무손상. v1/v2와 별도 아키텍처(state_dim 18).
실행:  py -3.12 -X utf8 train_v3.py
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {"h": 5.0, "b": 1000.0, "omega": 500.0, "CAP": 2000.0,
          "base_demand": 300.0, "days": 60, "tau_max": 7, "gamma": 1.0}
TRAIN_CONFIG = {"hidden_dim": 192, "supplier_emb_dim": 48, "num_layers": 4, "num_heads": 8}
SUPPLIERS = {
    0: {"name": "None", "tau": 0, "P": 0, "MOQ": 0, "C_cap": 1, "C_ship": 0},
    1: {"name": "A", "tau": 7, "P": 80, "MOQ": 1000, "C_cap": 2000, "C_ship": 5000},
    2: {"name": "B", "tau": 5, "P": 100, "MOQ": 500, "C_cap": 1000, "C_ship": 3000},
    3: {"name": "C", "tau": 3, "P": 120, "MOQ": 200, "C_cap": 1000, "C_ship": 2000},
    4: {"name": "D", "tau": 1, "P": 150, "MOQ": 50, "C_cap": 500, "C_ship": 5000},
    5: {"name": "E", "tau": 0, "P": 200, "MOQ": 0, "C_cap": 100, "C_ship": 2000},
}
SEAS_M = [1.0, 1.0, 0.9, 1.0, 1.1, 1.2, 1.5, 1.5, 1.0, 0.9, 1.1, 1.3]
SEAS_D = [1.2, 1.1, 1.0, 1.0, 0.9, 0.6, 0.5]
AVAIL_P = 0.55          # 각 업체(A~E) 당일 가용 확률
N_SUP = 5               # 마스킹 대상 업체 수 (A~E)


def demand_factors(t):
    def mu(tt):
        return SEAS_M[(tt // 30) % 12] * SEAS_D[tt % 7]
    return [mu(t), sum(mu(t + k) for k in range(1, 4)) / 3, sum(mu(t + k) for k in range(1, 8)) / 7]


# 상태 = 재고(1)+파이프라인(7)+시간(2)+수요예측(3)+가용성(5) = 18
STATE_DIM = 1 + CONFIG["tau_max"] + 2 + 3 + N_SUP

ITERS = 1000
BATCH = 256
LR = 1e-4
ENT_COEF_START = 0.04
ENT_COEF_END = 0.008
QTY_SIGMA_FLOOR = 30.0
QTY_BIAS_INIT = 500.0
LOG_EVERY = 20
SAVE_PATH = "retrained/best_model_v3.pt"
LOG_PATH = "train_v3.log"

LOGF = open(LOG_PATH, "w", encoding="utf-8")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); LOGF.write(line + "\n"); LOGF.flush()


def apply_mask(sup_logits, avail):
    """가용성 마스킹: 품절 업체(A~E) 로짓을 -inf. None(0)은 항상 허용."""
    masked = sup_logits.clone()
    masked[:, 1:1 + N_SUP] = torch.where(avail > 0.5, sup_logits[:, 1:1 + N_SUP],
                                         torch.full_like(sup_logits[:, 1:1 + N_SUP], -1e9))
    return masked


class AvailInventoryEnv:
    def __init__(self, config, batch_size):
        self.cfg = config
        self.batch_size = batch_size
        self.tau_max = config["tau_max"]
        self.gen = torch.Generator(device=DEVICE)

    def reset(self, seed):
        self.gen.manual_seed(seed)
        self.on_hand = torch.rand(self.batch_size, device=DEVICE, generator=self.gen) * self.cfg["CAP"]
        self.pipeline = torch.zeros(self.batch_size, self.tau_max + 1, device=DEVICE)
        self.t_global = torch.randint(0, 365 - self.cfg["days"], (1,), generator=self.gen, device=DEVICE).item()
        self.t_step = 0
        return self._get_state()

    def _sample_avail(self):
        r = torch.rand(self.batch_size, N_SUP, device=DEVICE, generator=self.gen)
        return (r < AVAIL_P).float()

    def _get_state(self):
        self.avail = self._sample_avail()  # 당일 가용성 (상태와 함께 고정)
        inv = self.on_hand.unsqueeze(1) / self.cfg["CAP"]
        pipe = self.pipeline[:, 1:] / self.cfg["CAP"]
        sin_t = math.sin(2 * math.pi * self.t_global / 365)
        cos_t = math.cos(2 * math.pi * self.t_global / 365)
        time_feat = torch.tensor([[sin_t, cos_t]], device=DEVICE).repeat(self.batch_size, 1)
        dem = torch.tensor([demand_factors(self.t_global)], device=DEVICE).repeat(self.batch_size, 1)
        return torch.cat([inv, pipe, time_feat, dem, self.avail], dim=1)

    def _get_demand(self):
        m = (self.t_global // 30) % 12; d = self.t_global % 7
        mu_t = self.cfg["base_demand"] * SEAS_M[m] * SEAS_D[d]
        noise = torch.randn(self.batch_size, device=DEVICE, generator=self.gen)
        return torch.clamp(mu_t + mu_t * 0.2 * noise, min=0).round()

    @torch.no_grad()
    def step(self, supplier_idx, order_qty):
        order_qty = order_qty.clone()
        cost_purch = torch.zeros(self.batch_size, device=DEVICE)
        cost_ship = torch.zeros(self.batch_size, device=DEVICE)
        instant = torch.zeros(self.batch_size, device=DEVICE)
        for s_code, s_info in SUPPLIERS.items():
            if s_code == 0:
                continue
            # 가용성 강제: 품절 업체 선택 시 무주문 처리(안전망; 마스킹으로 거의 발생 안 함)
            avail_mask = self.avail[:, s_code - 1] > 0.5
            mask = (supplier_idx == s_code) & avail_mask
            if mask.sum() == 0:
                continue
            corrected = torch.max(order_qty[mask], torch.tensor(float(s_info["MOQ"]), device=DEVICE))
            order_qty[mask] = corrected
            tau = s_info["tau"]
            if tau == 0:
                instant[mask] += corrected
            elif tau <= self.tau_max:
                self.pipeline[mask, tau] += corrected
            cost_purch[mask] += corrected * s_info["P"]
            cost_ship[mask] += torch.ceil(corrected / s_info["C_cap"]) * s_info["C_ship"]
        arrived = self.pipeline[:, 1].clone() + instant
        self.pipeline[:, :-1] = self.pipeline[:, 1:].clone()
        self.pipeline[:, -1] = 0
        self.on_hand += arrived
        self.on_hand -= self._get_demand()
        cost = (cost_purch + cost_ship + self.cfg["h"] * torch.relu(self.on_hand)
                + self.cfg["b"] * torch.relu(-self.on_hand)
                + self.cfg["omega"] * torch.relu(self.on_hand - self.cfg["CAP"]))
        self.t_global += 1; self.t_step += 1
        done = self.t_step >= self.cfg["days"]
        return self._get_state(), cost, done


class InventoryTransformer(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_layers=2, num_heads=4):
        super().__init__()
        self.input_embed = nn.Linear(state_dim, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                         dim_feedforward=256, batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.supplier_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 6))
        self.sup_action_embed = nn.Embedding(6, TRAIN_CONFIG["supplier_emb_dim"])
        self.qty_input_dim = hidden_dim + TRAIN_CONFIG["supplier_emb_dim"]
        self.qty_head_mu = nn.Sequential(nn.Linear(self.qty_input_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.qty_head_sigma = nn.Sequential(nn.Linear(self.qty_input_dim, 64), nn.ReLU(),
                                            nn.Linear(64, 1), nn.Softplus())
        self.qty_head_mu[2].bias.data.fill_(QTY_BIAS_INIT)

    def forward(self, state, fixed_supplier=None):
        x = self.input_embed(state).unsqueeze(1)
        feats = self.transformer(x).squeeze(1)
        sup_logits = self.supplier_head(feats)
        selected = torch.argmax(sup_logits, dim=1) if fixed_supplier is None else fixed_supplier
        sup_emb = self.sup_action_embed(selected)
        qty_input = torch.cat([feats, sup_emb], dim=1)
        mu = F.softplus(self.qty_head_mu(qty_input))
        sigma = self.qty_head_sigma(qty_input) + 1e-4
        return sup_logits, mu, sigma


def rollout(model, env, sample, seed):
    state = env.reset(seed)
    total_cost = torch.zeros(env.batch_size, device=DEVICE)
    logprob_sum = torch.zeros(env.batch_size, device=DEVICE)
    ent_sum = torch.zeros(env.batch_size, device=DEVICE)
    qty_acc, n = 0.0, 0
    done = False
    while not done:
        avail = state[:, -N_SUP:]
        sup_logits, _, _ = model(state)
        masked = apply_mask(sup_logits, avail)
        dist_s = Categorical(logits=masked)
        sup_idx = dist_s.sample() if sample else torch.argmax(masked, dim=1)
        _, mu, sigma = model(state, fixed_supplier=sup_idx)
        mu = mu.squeeze(-1)
        sigma = (sigma.squeeze(-1) + QTY_SIGMA_FLOOR) if sample else sigma.squeeze(-1)
        dist_q = Normal(mu, sigma)
        qty = torch.clamp(dist_q.sample() if sample else mu, min=0.0)
        if sample:
            logprob_sum = logprob_sum + dist_s.log_prob(sup_idx) + dist_q.log_prob(qty)
            ent_sum = ent_sum + dist_s.entropy()
        state, cost, done = env.step(sup_idx, qty)
        total_cost = total_cost + cost
        qty_acc += qty.mean().item(); n += 1
    return total_cost, logprob_sum, ent_sum / n, qty_acc / n


@torch.no_grad()
def evaluate(model, env, seeds=(0, 1, 2, 3)):
    model.eval()
    costs, usage = [], torch.zeros(6, device=DEVICE)
    for s in seeds:
        state = env.reset(s); total = torch.zeros(env.batch_size, device=DEVICE); done = False
        while not done:
            avail = state[:, -N_SUP:]
            logits, _, _ = model(state)
            sup = torch.argmax(apply_mask(logits, avail), dim=1)
            _, mu, _ = model(state, fixed_supplier=sup)
            for i in range(6):
                usage[i] += (sup == i).sum()
            state, cost, done = env.step(sup, mu.squeeze(-1))
            total = total + cost
        costs.append(total.mean().item())
    dist = (usage / usage.sum() * 100).tolist()
    return float(np.mean(costs)), dist


def main():
    torch.manual_seed(0); np.random.seed(0)
    log(f"DEVICE={DEVICE} ITERS={ITERS} BATCH={BATCH} STATE_DIM={STATE_DIM} avail_p={AVAIL_P}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    model = InventoryTransformer(STATE_DIM, TRAIN_CONFIG["hidden_dim"],
                                 TRAIN_CONFIG["num_layers"], TRAIN_CONFIG["num_heads"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    env = AvailInventoryEnv(CONFIG, BATCH)
    best_cost = float("inf"); t0 = time.time()
    for it in range(1, ITERS + 1):
        ent_coef = ENT_COEF_START + (ENT_COEF_END - ENT_COEF_START) * (it / ITERS)
        seed = 100000 + it
        model.train()
        c_s, logp, ent, qty_mean = rollout(model, env, True, seed)
        with torch.no_grad():
            c_g, _, _, _ = rollout(model, env, False, seed)
        adv = (c_s - c_g).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss = (adv * logp).mean() - ent_coef * ent.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if it % LOG_EVERY == 0 or it == 1:
            ec, ed = evaluate(model, env)
            if ec < best_cost:
                best_cost = ec; torch.save(model.state_dict(), SAVE_PATH)
            dists = " ".join(f"{SUPPLIERS[i]['name']}={ed[i]:.0f}%" for i in range(6))
            eta = (time.time() - t0) / it * (ITERS - it)
            log(f"it{it:>4}/{ITERS} | eval={ec:>11,.0f} best={best_cost:>11,.0f} "
                f"| ent={ent.mean().item():.3f} qty~{qty_mean:.0f} | [{dists}] | ETA {eta/60:.1f}m")
    log(f"DONE best={best_cost:,.0f} saved={SAVE_PATH} ({(time.time()-t0)/60:.1f}m)")
    LOGF.close()


if __name__ == "__main__":
    main()
