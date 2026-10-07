"""공정 벤치마크 재현 스크립트 (2026-10-07 작성).

README·POLICY_ANALYSIS.md의 재검증 수치(단일 최적 base-stock 8.93M, 구 체크포인트 12.64M,
재학습판 7.57M / 200일·시드 8개)를 낸 당시 평가 코드가 남아 있지 않아 다시 작성했다.

측정 내용
  1. 업체 A~E 각각을 단일 공급원으로 쓰는 base-stock 정책의 최적 S*를 그리드 서치
  2. 구 체크포인트(models/best_model_epoch_049_cost_32.pt)의 greedy 정책 비용
  3. 재학습판(serve/retrained/best_model_retrained.pt)의 greedy 정책 비용

조건: 에피소드 200일, 시드 0~7, 시드마다 BATCH개 시나리오(초기재고·시작일·수요)를 공유.
      같은 시드면 모든 정책이 같은 시나리오를 본다(공통 난수).
환경은 serve/train.py의 VectorizedInventoryEnv와 같은 동역학이며 상태는 v1 형식(10차원)이다.

당시 코드의 배치 크기와 난수 사용 순서를 알 수 없어 수치가 자릿수까지 일치하지는 않는다.
비교할 것은 순위와 비율이다.

실행:  python serve/benchmark.py            (repo 루트에서)
"""
import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {"h": 5.0, "b": 1000.0, "omega": 500.0, "CAP": 2000.0,
          "base_demand": 300.0, "tau_max": 7}
# v1 아키텍처 (구 체크포인트와 재학습판이 공유)
ARCH = {"hidden_dim": 128, "supplier_emb_dim": 32, "num_layers": 3, "num_heads": 8}
STATE_DIM = 1 + CONFIG["tau_max"] + 2
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


class Env:
    """serve/train.py의 VectorizedInventoryEnv와 같은 동역학. 상태만 v1(10차원)."""

    def __init__(self, days, batch):
        self.days, self.batch, self.tau_max = days, batch, CONFIG["tau_max"]
        self.gen = torch.Generator(device=DEVICE)

    def reset(self, seed):
        self.gen.manual_seed(seed)
        self.on_hand = torch.rand(self.batch, device=DEVICE, generator=self.gen) * CONFIG["CAP"]
        self.pipeline = torch.zeros(self.batch, self.tau_max + 1, device=DEVICE)
        self.t_global = torch.randint(0, 365 - self.days, (1,), generator=self.gen, device=DEVICE).item()
        self.t_step = 0
        return self.state()

    def state(self):
        inv = self.on_hand.unsqueeze(1) / CONFIG["CAP"]
        pipe = self.pipeline[:, 1:] / CONFIG["CAP"]
        ang = 2 * math.pi * self.t_global / 365
        tf = torch.tensor([[math.sin(ang), math.cos(ang)]], device=DEVICE).repeat(self.batch, 1)
        return torch.cat([inv, pipe, tf], dim=1)

    def demand(self):
        mu = CONFIG["base_demand"] * SEAS_M[(self.t_global // 30) % 12] * SEAS_D[self.t_global % 7]
        noise = torch.randn(self.batch, device=DEVICE, generator=self.gen)
        return torch.clamp(mu + mu * 0.2 * noise, min=0).round()

    def step(self, sup, qty):
        qty = qty.clone()
        cost = torch.zeros(self.batch, device=DEVICE)
        instant = torch.zeros(self.batch, device=DEVICE)
        for code, s in SUPPLIERS.items():
            if code == 0:
                continue
            m = sup == code
            if not m.any():
                continue
            q = torch.clamp(qty[m], min=float(s["MOQ"]))  # MOQ 보정
            if s["tau"] == 0:
                instant[m] += q
            else:
                self.pipeline[m, s["tau"]] += q
            cost[m] += q * s["P"] + torch.ceil(q / s["C_cap"]) * s["C_ship"]
        arrived = self.pipeline[:, 1].clone() + instant
        self.pipeline[:, :-1] = self.pipeline[:, 1:].clone()
        self.pipeline[:, -1] = 0
        self.on_hand = self.on_hand + arrived - self.demand()
        cost += (CONFIG["h"] * torch.relu(self.on_hand)
                 + CONFIG["b"] * torch.relu(-self.on_hand)
                 + CONFIG["omega"] * torch.relu(self.on_hand - CONFIG["CAP"]))
        self.t_global += 1
        self.t_step += 1
        return self.state(), cost, self.t_step >= self.days


class InventoryTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        h = ARCH["hidden_dim"]
        self.input_embed = nn.Linear(STATE_DIM, h)
        enc = nn.TransformerEncoderLayer(d_model=h, nhead=ARCH["num_heads"],
                                         dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=ARCH["num_layers"])
        self.supplier_head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 6))
        self.sup_action_embed = nn.Embedding(6, ARCH["supplier_emb_dim"])
        qin = h + ARCH["supplier_emb_dim"]
        self.qty_head_mu = nn.Sequential(nn.Linear(qin, 64), nn.ReLU(), nn.Linear(64, 1))
        self.qty_head_sigma = nn.Sequential(nn.Linear(qin, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

    @torch.no_grad()
    def act(self, state):
        feats = self.transformer(self.input_embed(state).unsqueeze(1)).squeeze(1)
        sup = torch.argmax(self.supplier_head(feats), dim=1)
        mu = F.softplus(self.qty_head_mu(torch.cat([feats, self.sup_action_embed(sup)], dim=1)))
        return sup, mu.squeeze(-1)


def load(path):
    model = InventoryTransformer().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(ROOT, path), map_location=DEVICE))
    return model.eval()


def run(policy, days, batch, seeds):
    """policy(env, state) -> (supplier_idx, qty). 시드별 평균 총비용과 업체 사용 횟수를 돌려준다."""
    env = Env(days, batch)
    per_seed, usage = [], torch.zeros(6)
    for seed in seeds:
        state, total, done = env.reset(seed), torch.zeros(batch, device=DEVICE), False
        while not done:
            sup, qty = policy(env, state)
            state, cost, done = env.step(sup, qty)
            total += cost
            usage += torch.bincount(sup.cpu(), minlength=6).float()
        per_seed.append(total.mean().item())
    return per_seed, usage


def base_stock(code, S):
    def policy(env, _state):
        ip = env.on_hand + env.pipeline.sum(dim=1)  # 재고 포지션
        need = ip < S
        sup = torch.where(need, torch.tensor(code, device=DEVICE), torch.tensor(0, device=DEVICE))
        return sup, torch.where(need, S - ip, torch.zeros_like(ip))
    return policy


def model_policy(model):
    return lambda env, state: model.act(state)


def fmt_usage(u):
    p = u / u.sum() * 100
    return " ".join(f"{SUPPLIERS[i]['name']}={p[i]:.0f}%" for i in range(6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--grid-step", type=int, default=100)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    mean = lambda xs: sum(xs) / len(xs)
    print(f"DEVICE={DEVICE} days={a.days} batch={a.batch} seeds={seeds}")

    print("\n[1] 단일 업체 base-stock — S* 그리드 서치")
    best = {}
    for code in range(1, 6):
        results = []
        for S in range(0, 5001, a.grid_step):
            per_seed, _ = run(base_stock(code, float(S)), a.days, a.batch, seeds)
            results.append((mean(per_seed), S, per_seed))
        cost, S, per_seed = min(results)
        best[code] = (cost, S, per_seed)
        print(f"  {SUPPLIERS[code]['name']}: 최저 총비용 {cost:>13,.0f}  S*={S}")
    bc = min(best, key=lambda c: best[c][0])
    b_cost, b_S, b_seed = best[bc]
    print(f"  → 단일 최적 = {SUPPLIERS[bc]['name']} (S*={b_S}) {b_cost:,.0f}")

    print("\n[2] 학습 정책 (greedy)")
    rows = {}
    for label, path in [("구 체크포인트", "models/best_model_epoch_049_cost_32.pt"),
                        ("재학습판", "serve/retrained/best_model_retrained.pt")]:
        per_seed, usage = run(model_policy(load(path)), a.days, a.batch, seeds)
        rows[label] = (mean(per_seed), per_seed)
        print(f"  {label}: {mean(per_seed):>13,.0f}  사용[{fmt_usage(usage)}]")

    old, new = rows["구 체크포인트"][0], rows["재학습판"][0]
    print("\n[3] 비교")
    print(f"  구 체크포인트 / 단일 최적 = {old / b_cost:.2f}배")
    print(f"  재학습판 vs 단일 최적     = {(1 - new / b_cost) * 100:+.1f}% 절감")
    print(f"  재학습판 vs 구 체크포인트 = {(1 - new / old) * 100:+.1f}% 절감")
    wins = sum(n < b for n, b in zip(rows["재학습판"][1], b_seed))
    print(f"  시드별 승패 (재학습판 < 단일 최적): {wins}/{len(seeds)}")
    for s, (n, b) in enumerate(zip(rows["재학습판"][1], b_seed)):
        print(f"    seed {s}: 재학습판 {n:>12,.0f} | 단일 최적 {b:>12,.0f} | {(1 - n / b) * 100:+.1f}%")


if __name__ == "__main__":
    main()
