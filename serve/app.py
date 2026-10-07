"""발주 추천 추론 API (v3 — 가용성 인지).

재고 상태 + 당일 업체 가용성을 받아 RL 에이전트의 발주 추천을 반환한다.
실행:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import CONFIG, N_SUP, SUPPLIERS, demand_factors, load_model, recommend

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model_v3.pt")

app = FastAPI(title="Dynamic Supplier Selection API", version="3.0.0")
model = None


@app.on_event("startup")
def _startup():
    global model
    model = load_model(MODEL_PATH)


class PredictRequest(BaseModel):
    on_hand: float = Field(..., description="현재 보유 재고", examples=[1000.0])
    pipeline: list[float] = Field(..., description="향후 1..7일 입고 예정 (길이 7)",
                                  examples=[[0, 0, 0, 0, 0, 0, 0]])
    day_of_year: int = Field(0, ge=0, le=364, description="연중 일자 0~364 (계절성)")
    available: list[bool] = Field(..., description="당일 A~E 업체 가용 여부 (길이 5)",
                                  examples=[[True, True, True, True, True]])


class PredictResponse(BaseModel):
    supplier_idx: int
    supplier_name: str
    order_qty: int
    detail: dict


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.pipeline) != CONFIG["tau_max"]:
        raise HTTPException(422, f"pipeline 길이 {CONFIG['tau_max']} 필요 (받음 {len(req.pipeline)})")
    if len(req.available) != N_SUP:
        raise HTTPException(422, f"available 길이 {N_SUP} 필요 (받음 {len(req.available)})")
    idx, name, qty = recommend(model, req.on_hand, req.pipeline, req.day_of_year, req.available)
    info = SUPPLIERS[idx]
    return PredictResponse(
        supplier_idx=idx, supplier_name=name, order_qty=qty,
        detail={"lead_time": info["tau"], "unit_price": info["P"], "moq": info["MOQ"],
                "demand_forecast": [round(x, 2) for x in demand_factors(req.day_of_year)]},
    )
