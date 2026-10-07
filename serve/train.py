"""REINFORCE 재학습 트레이너 (정책붕괴 + 주문량head 사망 수정판).

원본 repo에 학습루프가 유실되어 재작성. 환경/모델 아키텍처는 원본과 동일 유지
(새 체크포인트가 serve/model.py 추론코드에 그대로 로드되도록).

핵심 수정:
  1. qty_head_mu bias 초기값 ↑ (5 → ~400)
     - MOQ 클램핑 때문에 raw_qty < MOQ 구간은 보상 불변 → gradient 0 → 학습불가였음.
       시작 mu를 MOQ 위로 올려 gradient 흐르게 함.
  2. 엔트로피 정규화 (조기 정책붕괴 차단, 전 업체 탐색 보장)
  3. Greedy rollout baseline (self-critic) + advantage 정규화 (분산↓)

진행률: stdout + serve/train.log 에 주기적 로그 (업체사용분포/비용/엔트로피/주문량).
실행:  py -3.12 -X utf8 train.py
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# ==========================================
# 0. Setup & Hyperparams
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "h": 5.0, "b": 1000.0, "omega": 500.0, "CAP": 2000.0,
    "base_demand": 300.0, "days": 60, "tau_max": 7, "gamma": 1.0,
}
TRAIN_CONFIG = {"hidden_dim": 192, "supplier_emb_dim": 48, "num_layers": 4, "num_heads": 8}
SUPPLIERS = {
    0: {"name": "None", "tau": 0, "P": 0, "MOQ": 0, "C_cap": 1, "C_ship": 0},
    1: {"name": "A", "tau": 7, "P": 80, "MOQ": 1000, "C_cap": 2000, "C_ship": 5000},
    2: {"name": "B", "tau": 5, "P": 100, "MOQ": 500, "C_cap": 1000, "C_ship": 3000},
    3: {"name": "C", "tau": 3, "P": 120, "MOQ": 200, "C_cap": 1000, "C_ship": 2000},
    4: {"name": "D", "tau": 1, "P": 150, "MOQ": 50, "C_cap": 500, "C_ship": 5000},
    5: {"name": "E", "tau": 0, "P": 200, "MOQ": 0, "C_cap": 100, "C_ship": 2000},
}
# 계절성 (env·feature 공유). 값 = base_demand 대비 배수.
SEAS_M = [1.0, 1.0, 0.9, 1.0, 1.1, 1.2, 1.5, 1.5, 1.0, 0.9, 1.1, 1.3]
SEAS_D = [1.2, 1.1, 1.0, 1.0, 0.9, 0.6, 0.5]


def demand_factors(t):
    """기대수요 배수 [오늘, 향후3일평균, 향후7일평균]. 상태에 명시해 상황민감도↑.
    A(리드7)는 7일전망, C/D(리드1~3)는 3일전망 보고 판단 가능."""
    def mu(tt):
        return SEAS_M[(tt // 30) % 12] * SEAS_D[tt % 7]
    f0 = mu(t)
    f3 = sum(mu(t + k) for k in range(1, 4)) / 3
    f7 = sum(mu(t + k) for k in range(1, 8)) / 7
    return [f0, f3, f7]


# 상태 = 재고(1) + 파이프라인(tau_max) + 시간(sin,cos=2) + 수요예측(3)
STATE_DIM = 1 + CONFIG["tau_max"] + 2 + 3

# 학습 하이퍼파라미터
ITERS = 1200
BATCH = 256
LR = 1e-4
ENT_COEF_START = 0.03
ENT_COEF_END = 0.005
QTY_SIGMA_FLOOR = 30.0      # 주문량 탐색용 최소 표준편차
QTY_BIAS_INIT = 500.0       # mu 시작값을 MOQ 위로 (A의 MOQ1000 헤드룸 고려)
LOG_EVERY = 20
SAVE_PATH = "retrained/best_model_v2.pt"
LOG_PATH = "train_v2.log"

LOGF = open(LOG_PATH, "w", encoding="utf-8")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOGF.write(line + "\n"); LOGF.flush()


# ==========================================
# 1. Vectorized Environment (원본 동일)
# ==========================================
class VectorizedInventoryEnv:
    def __init__(self, config, batch_size):
        self.cfg = config
        self.batch_size = batch_size
        self.tau_max = config["tau_max"]
        self.seasonal_month = torch.tensor(
            [1.0, 1.0, 0.9, 1.0, 1.1, 1.2, 1.5, 1.5, 1.0, 0.9, 1.1, 1.3], device=DEVICE)
        self.seasonal_day = torch.tensor([1.2, 1.1, 1.0, 1.0, 0.9, 0.6, 0.5], device=DEVICE)
        # 시나리오 전용 generator (초기재고·시작일·수요). 행동샘플링(global RNG)과 분리.
        # → sample/greedy 롤아웃을 같은 seed로 돌리면 완전 동일 시나리오 = 페어드 베이스라인.
        self.gen = torch.Generator(device=DEVICE)

    def reset(self, seed):
        self.gen.manual_seed(seed)
        self.on_hand = torch.rand(self.batch_size, device=DEVICE, generator=self.gen) * self.cfg["CAP"]
        self.pipeline = torch.zeros(self.batch_size, self.tau_max + 1, device=DEVICE)
        self.t_global = torch.randint(0, 365 - self.cfg["days"], (1,),
                                      generator=self.gen, device=DEVICE).item()
        self.t_step = 0
        return self._get_state()

    def _get_state(self):
        inv_feat = self.on_hand.unsqueeze(1) / self.cfg["CAP"]
        pipe_feat = self.pipeline[:, 1:] / self.cfg["CAP"]
        sin_t = math.sin(2 * math.pi * self.t_global / 365)
        cos_t = math.cos(2 * math.pi * self.t_global / 365)
        time_feat = torch.tensor([[sin_t, cos_t]], device=DEVICE).repeat(self.batch_size, 1)
        dem_feat = torch.tensor([demand_factors(self.t_global)], device=DEVICE).repeat(self.batch_size, 1)
        return torch.cat([inv_feat, pipe_feat, time_feat, dem_feat], dim=1)

    def _get_demand(self):
        m = (self.t_global // 30) % 12
        d = self.t_global % 7
        mu_t = self.cfg["base_demand"] * self.seasonal_month[m] * self.seasonal_day[d]
        sigma_t = mu_t * 0.2
        noise = torch.randn(self.batch_size, device=DEVICE, generator=self.gen)
        demand = mu_t + sigma_t * noise
        return torch.clamp(demand, min=0).round()

    @torch.no_grad()
    def step(self, supplier_idx, order_qty):
        order_qty = order_qty.clone()
        cost_purch = torch.zeros(self.batch_size, device=DEVICE)
        cost_ship = torch.zeros(self.batch_size, device=DEVICE)
        instant_arrival = torch.zeros(self.batch_size, device=DEVICE)
        for s_code, s_info in SUPPLIERS.items():
            if s_code == 0:
                continue
            mask = supplier_idx == s_code
            if mask.sum() == 0:
                continue
            corrected = torch.max(order_qty[mask],
                                  torch.tensor(float(s_info["MOQ"]), device=DEVICE))
            order_qty[mask] = corrected
            tau = s_info["tau"]
            if tau == 0:
                instant_arrival[mask] += corrected
            elif tau <= self.tau_max:
                self.pipeline[mask, tau] += corrected
            cost_purch[mask] += corrected * s_info["P"]
            cost_ship[mask] += torch.ceil(corrected / s_info["C_cap"]) * s_info["C_ship"]
        arrived = self.pipeline[:, 1].clone() + instant_arrival
        self.pipeline[:, :-1] = self.pipeline[:, 1:].clone()
        self.pipeline[:, -1] = 0
        self.on_hand += arrived
        self.on_hand -= self._get_demand()
        cost_hold = self.cfg["h"] * torch.relu(self.on_hand)
        cost_short = self.cfg["b"] * torch.relu(-self.on_hand)
        cost_over = self.cfg["omega"] * torch.relu(self.on_hand - self.cfg["CAP"])
        total_cost = cost_purch + cost_ship + cost_hold + cost_short + cost_over
        self.t_global += 1
        self.t_step += 1
        done = self.t_step >= self.cfg["days"]
        return self._get_state(), total_cost, done


# ==========================================
# 2. Model (아키텍처 원본 동일, bias 초기값만 수정)
# ==========================================
class InventoryTransformer(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_layers=2, num_heads=4):
        super().__init__()
        self.input_embed = nn.Linear(state_dim, hidden_dim)
        # dropout=0: forward 결정성 보장 (페어드 베이스라인 재현성 + greedy 일관성).
        # 추론(model.py)은 eval()로 어차피 dropout off → 체크포인트 호환.
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                         dim_feedforward=256, batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.supplier_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 6))
        self.sup_action_embed = nn.Embedding(6, TRAIN_CONFIG["supplier_emb_dim"])
        self.qty_input_dim = hidden_dim + TRAIN_CONFIG["supplier_emb_dim"]
        self.qty_head_mu = nn.Sequential(
            nn.Linear(self.qty_input_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.qty_head_sigma = nn.Sequential(
            nn.Linear(self.qty_input_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())
        self.qty_head_mu[2].bias.data.fill_(QTY_BIAS_INIT)  # 수정1

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
    """한 에피소드(batch) 롤아웃.
    sample=True: 확률 샘플링(학습용, grad), False: greedy(baseline, no grad).
    seed: 시나리오 고정용 — sample/greedy 같은 seed = 동일 수요/초기재고 (페어드 베이스라인)."""
    state = env.reset(seed)
    total_cost = torch.zeros(env.batch_size, device=DEVICE)
    logprob_sum = torch.zeros(env.batch_size, device=DEVICE)
    ent_sum = torch.zeros(env.batch_size, device=DEVICE)
    usage = torch.zeros(6, device=DEVICE)
    qty_acc, n_steps = 0.0, 0
    done = False
    while not done:
        sup_logits, _, _ = model(state)
        dist_s = Categorical(logits=sup_logits)
        if sample:
            sup_idx = dist_s.sample()
        else:
            sup_idx = torch.argmax(sup_logits, dim=1)
        _, mu, sigma = model(state, fixed_supplier=sup_idx)
        mu = mu.squeeze(-1)
        sigma = (sigma.squeeze(-1) + QTY_SIGMA_FLOOR) if sample else sigma.squeeze(-1)
        dist_q = Normal(mu, sigma)
        qty = torch.clamp(dist_q.sample() if sample else mu, min=0.0)
        if sample:
            lp = dist_s.log_prob(sup_idx) + dist_q.log_prob(qty)
            logprob_sum = logprob_sum + lp
            ent_sum = ent_sum + dist_s.entropy()
        state, cost, done = env.step(sup_idx, qty)
        total_cost = total_cost + cost
        for i in range(6):
            usage[i] += (sup_idx == i).sum()
        qty_acc += qty.mean().item(); n_steps += 1
    return total_cost, logprob_sum, ent_sum / n_steps, usage, qty_acc / n_steps


@torch.no_grad()
def evaluate(model, env, seeds=(0, 1, 2, 3)):
    """고정 시나리오서 greedy 정책 평가 (체크포인트 선택용, 편향없는 비교)."""
    model.eval()
    costs, usage = [], torch.zeros(6, device=DEVICE)
    for s in seeds:
        c, _, _, u, _ = rollout(model, env, sample=False, seed=s)
        costs.append(c.mean().item()); usage += u
    dist = (usage / usage.sum() * 100).tolist()
    return float(np.mean(costs)), dist


def main():
    torch.manual_seed(0); np.random.seed(0)
    log(f"DEVICE={DEVICE} | ITERS={ITERS} BATCH={BATCH} LR={LR}")
    log(f"qty_bias_init={QTY_BIAS_INIT} sigma_floor={QTY_SIGMA_FLOOR} ent={ENT_COEF_START}->{ENT_COEF_END}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    model = InventoryTransformer(STATE_DIM, TRAIN_CONFIG["hidden_dim"],
                                 TRAIN_CONFIG["num_layers"], TRAIN_CONFIG["num_heads"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    env = VectorizedInventoryEnv(CONFIG, BATCH)
    best_cost = float("inf")
    t0 = time.time()

    for it in range(1, ITERS + 1):
        ent_coef = ENT_COEF_START + (ENT_COEF_END - ENT_COEF_START) * (it / ITERS)
        seed = 100000 + it  # 이 iter의 시나리오 (sample/greedy 공유)
        # 샘플 롤아웃 (grad)
        model.train()
        c_sample, logp, ent, usage, qty_mean = rollout(model, env, sample=True, seed=seed)
        # greedy baseline (no grad) — 동일 시나리오
        with torch.no_grad():
            c_greedy, _, _, usage_g, _ = rollout(model, env, sample=False, seed=seed)
        # advantage (cost 기준, 낮을수록 좋음) + 정규화
        adv = (c_sample - c_greedy).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss_pg = (adv * logp).mean()
        loss = loss_pg - ent_coef * ent.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % LOG_EVERY == 0 or it == 1:
            eval_cost, eval_dist = evaluate(model, env)
            if eval_cost < best_cost:
                best_cost = eval_cost
                torch.save(model.state_dict(), SAVE_PATH)
            dists = " ".join(f"{SUPPLIERS[i]['name']}={eval_dist[i]:.0f}%" for i in range(6))
            eta = (time.time() - t0) / it * (ITERS - it)
            log(f"it{it:>4}/{ITERS} | eval_cost={eval_cost:>11,.0f} "
                f"best={best_cost:>11,.0f} | ent={ent.mean().item():.3f} qty~{qty_mean:.0f} "
                f"| 사용[{dists}] | ETA {eta/60:.1f}m")

    log(f"DONE best_eval_cost={best_cost:,.0f} saved={SAVE_PATH} ({(time.time()-t0)/60:.1f}m)")
    LOGF.close()


if __name__ == "__main__":
    main()
