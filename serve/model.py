"""추론 전용 모델 정의 — v3 (가용성 인지 컨틴전시 정책).

상태 = 재고(1) + 파이프라인(7) + 시간 sin/cos(2) + 수요예측(3) + 업체가용성(5) = 18차원.
매일 일부 업체가 품절될 수 있으며, 정책은 가용 업체 중 상황별 최적을 선택한다.
학습/환경 로직은 train_v3.py 참조.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")  # 배포는 CPU 추론

CONFIG = {"CAP": 2000.0, "tau_max": 7}
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
N_SUP = 5  # 가용성 마스킹 대상 (A~E)

STATE_DIM = 1 + CONFIG["tau_max"] + 2 + 3 + N_SUP  # = 18


def demand_factors(t):
    def mu(tt):
        return SEAS_M[(tt // 30) % 12] * SEAS_D[tt % 7]
    return [mu(t), sum(mu(t + k) for k in range(1, 4)) / 3, sum(mu(t + k) for k in range(1, 8)) / 7]


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


def apply_mask(sup_logits, avail):
    """품절 업체(A~E) 로짓을 -inf. None(0)은 항상 허용."""
    masked = sup_logits.clone()
    masked[:, 1:1 + N_SUP] = torch.where(avail > 0.5, sup_logits[:, 1:1 + N_SUP],
                                         torch.full_like(sup_logits[:, 1:1 + N_SUP], -1e9))
    return masked


def apply_moq(supplier_idx: int, raw_qty: float) -> int:
    """모델 raw 주문량 → 실제 발주량 (MOQ 보정)."""
    if supplier_idx == 0:
        return 0
    return int(round(max(raw_qty, float(SUPPLIERS[supplier_idx]["MOQ"]))))


def build_state(on_hand, pipeline, day_of_year, available):
    """원시 입력 → 상태 텐서 (1, 18).

    on_hand     : 현재 보유 재고
    pipeline    : 향후 1..7일 입고 예정 (길이 7)
    day_of_year : 0~364
    available   : A~E 가용 여부 (길이 5, 1/0 또는 bool)
    """
    cap = CONFIG["CAP"]; tau_max = CONFIG["tau_max"]
    if len(pipeline) != tau_max:
        raise ValueError(f"pipeline 길이 {tau_max} 필요 (받음 {len(pipeline)})")
    if len(available) != N_SUP:
        raise ValueError(f"available 길이 {N_SUP} 필요 (받음 {len(available)})")
    sin_t = math.sin(2 * math.pi * day_of_year / 365)
    cos_t = math.cos(2 * math.pi * day_of_year / 365)
    feat = ([on_hand / cap] + [p / cap for p in pipeline] + [sin_t, cos_t]
            + demand_factors(day_of_year) + [float(a) for a in available])
    return torch.tensor([feat], dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def recommend(model, on_hand, pipeline, day_of_year, available):
    """가용성 마스킹 + 결정적 선택 + MOQ 보정. 반환: (idx, name, qty)."""
    state = build_state(on_hand, pipeline, day_of_year, available)
    sup_logits, _, _ = model(state)
    masked = apply_mask(sup_logits, state[:, -N_SUP:])
    idx = int(torch.argmax(masked, dim=1).item())
    _, mu, _ = model(state, fixed_supplier=torch.tensor([idx], device=DEVICE))
    qty = apply_moq(idx, mu.item())
    return idx, SUPPLIERS[idx]["name"], qty


def load_model(model_path: str) -> InventoryTransformer:
    model = InventoryTransformer(STATE_DIM, TRAIN_CONFIG["hidden_dim"],
                                 TRAIN_CONFIG["num_layers"], TRAIN_CONFIG["num_heads"]).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model
