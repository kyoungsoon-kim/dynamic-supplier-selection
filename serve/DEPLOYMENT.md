# 🚀 배포 진행상황 (Dynamic Supplier Selection)

> 학습된 RL 에이전트(`models/best_model_epoch_049_cost_32.pt`)를 추론 서비스로 배포한 기록.
> 원본 repo는 **무수정**. 배포 산출물은 전부 `serve/` 하위 신규 생성.
> 최종 업데이트: 2026-06-15
> **2026-10-07 21:20 공개 시 추가**: 이 문서는 작업 당시의 기록입니다. 이후 루트 `README.md`에 재검증 결과와 배포 절을 반영했습니다. `serve/hf_space/`(Space에 올라가 있는 묶음), `retrained/best_model_v2.pt`(실패 실험), `train.log`(콘솔 로그와 같은 내용)는 저장소에 포함하지 않았습니다. 학습 로그는 `train_console.log` · `train_v2_console.log` · `train_v3_console.log`입니다.

---

## ✅ 완료 (STAGE 1~2, 4)

| 단계 | 내용 | 상태 |
| --- | --- | --- |
| STAGE 1 | 추론 API (FastAPI) | ✅ |
| STAGE 2 | Docker 포장 + 로컬 컨테이너 실행 | ✅ |
| STAGE 4 | Hugging Face Spaces 배포 (Gradio) | ✅ **라이브** |

### 🌐 라이브 데모
- **HF Space**: https://huggingface.co/spaces/ksk00/dynamic-supplier-selection
- 계정: `ksk00`
- SDK: Gradio 5.9.1 / CPU 무료 / 빌드~90초

---

## 📂 만든 파일 (전부 신규)

```
serve/
├ DEPLOYMENT.md          ← 이 문서
├ model.py               추론 전용 모델구조 + build_state() + load_model()
├ app.py                 FastAPI: POST /predict, GET /health
├ requirements.txt       FastAPI용 의존성
├ Dockerfile             CPU torch 기반 이미지 레시피
└ hf_space/              HF Space 업로드 묶음
    ├ app.py             Gradio UI (재고 입력 → 발주추천)
    ├ model.py           (serve/model.py 복사본)
    ├ requirements.txt   torch CPU + gradio
    ├ README.md          HF Space frontmatter 포함
    └ best_model_*.pt    가중치 (models/ 에서 복사, 1.7MB)
.dockerignore            (repo 루트)
```

원본 `src/`, `models/`, `notebooks/`, `README.md` → **건드리지 않음**.

---

## 🧠 모델 입출력 (배포 인터페이스)

- **입력 상태** = 재고(1) + 파이프라인(7, 향후 1~7일 입고예정) + 시간(sin,cos = 2) → **10차원**
  - 내부서 `on_hand/CAP`, `pipeline/CAP` 정규화 (`build_state()`)
- **출력 행동** = 공급업체 인덱스(0~5, 0=발주안함) + 주문량
- 추론 = `act_deterministic()` (argmax 공급업체 + 평균 주문량, 탐색 없음 = 원본 `deterministic=True` 재현)

---

## 🔁 재배포 / 운영 명령어

### HF Space 업데이트 (코드 수정 후)
```bash
# 파일 수정 → serve/hf_space/ 갱신 후
py -3.14 -X utf8 -c "from huggingface_hub import upload_folder; upload_folder(repo_id='ksk00/dynamic-supplier-selection', repo_type='space', folder_path='serve/hf_space', commit_message='update')"
```
> ⚠️ `serve/model.py` 고치면 `serve/hf_space/model.py`에도 복사 반영 필요 (둘은 별개 파일).

### 로컬 Docker (STAGE 2)
```bash
# 빌드 (build context = repo 루트)
docker build -f serve/Dockerfile -t dss-api .
# 실행
docker run -d --name dss -p 8000:8000 dss-api
# 테스트
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"on_hand":300,"pipeline":[0,0,0,0,0,0,0],"day_of_year":100,"available":[true,true,true,true,true]}'
# Swagger UI:  http://localhost:8000/docs
# 정리
docker rm -f dss
```

---

## 🔍 점검 기록 (2026-06-15): "출력 B/3 고정" 원인

증상: HF 데모가 늘 `B / 3단위` 추천 → 점검.

- 입력 200조합 스윕: 출력이 `E/5`(130) · `B/3`(64)로 **거의 2값 붕괴**
- 원본 환경 롤아웃 분석 → **핵심 발견**:
  - 모델 raw 주문량(~3)은 **무의미값**. 모델은 "**어느 공급업체**"만 학습
  - 실제 발주량은 환경 `step()`의 **MOQ 보정**(`max(raw, MOQ)`)이 결정
  - 즉 B 선택 = 500개(B의 MOQ), E 선택 = 5개
- **배포코드는 정상**(모델 raw 출력 충실 재현)이었으나 **MOQ 보정 누락**으로 무의미값 표시됨
- **조치**: `model.apply_moq()` 추가 → `app.py`·`hf_space/app.py` 적용 → 재배포 완료
  - 결과: 재고 0~800 → **B/500**, 재고 1500+ → **E/5** (의미있는 발주량)

> 참고: 공급업체 선택이 좁은 입력범위서 잘 안 바뀌는 건 모델 정책이 narrow하게 수렴한 것(REINFORCE 특성). 배포버그 아님.

---

## 🔁 재학습 (Plan A) — 정책붕괴 해소 ✅

`serve/train.py` 로 REINFORCE 트레이너 재작성(원본 학습루프 유실됨) → 재학습 → 재배포 완료.

### 수정 포인트
1. **qty_head bias 5→400**: MOQ 클램핑이 raw<MOQ 구간 gradient를 죽여 주문량head 학습불가였음. 시작 mu를 MOQ 위로 → gradient 부활
2. **엔트로피 정규화**(0.03→0.005 anneal): 조기 정책붕괴 차단
3. **Greedy rollout 페어드 베이스라인**: sample/greedy 동일 시나리오(전용 generator seed)로 advantage 분산↓
4. **dropout=0**: forward 결정성(베이스라인 재현성)

### 결과 (200일·8seed 동일틀)
| 정책 | 총비용 | 사용 업체 |
| --- | --- | --- |
| 구 체크포인트 | 12.64M | B·E (붕괴) |
| **재학습판** | **7.57M** | **B·C·D·None (다양)** |
| 단일최적 base-stock(C) | 8.93M | C |

→ **구 대비 40.1%↓, 단일최적 대비 15.2%↓.** 죽었던 C·D 부활, 최악 E·고비용 A 자동탈락.
체크포인트: `serve/retrained/best_model_retrained.pt` (HF Space 배포 중).
학습로그: `serve/train.log` (600 iter, 13분, RTX 3060 Ti).

---

## 🧠 정책 복잡도 향상 (v2 실패 → v3 성공)

목표: 상황별로 날카롭게 다른 행동을 내는 **상태민감·고복잡 정책**.

### v2 (실패) — feature·용량·학습량 ↑
수요예측 feature(state 10→13) + 모델용량↑(hidden 192·layer 4) + 1200 iter.
- 결과: 200일 9.02M (v1 7.57M보다 **나쁨**), 상태민감도 동일(4행동), 엔트로피 0.07로 **더 경직**
- 교훈: 이 환경은 **최적정책이 원래 단순**(단일 base-stock에 근접). 모델 손질론 복잡도 못 만듦.
- 미배포. 체크포인트 `retrained/best_model_v2.pt` (실험 보존), `train_v2.log`.

### v3 (성공) — 환경 재설계: 확률적 업체 가용성
`train_v3.py` + `AvailInventoryEnv`. 매일 각 업체 55% 확률 가용, 가용성을 상태에 노출 +
액션 마스킹. 정책이 "가용 업체 중 상황별 최적" 컨틴전시 플랜을 학습하도록 **복잡도를 환경이 강제**.
- 상태 = 재고+파이프라인+시간+수요예측+**가용성(5)** = 18차원
- 결과: **A~E 전 업체 활용**, 3축(재고긴급도·계절·가용성) 모두 반응
  - 저재고 → D(리드1), 고재고 → A(저가대량), 고시즌중간 → C
  - 가용성 폴백: 전부가용 A → A품절 C → ABC품절 D → E만 None
- 상태민감도 6행동(v1·v2 = 4), 업체사용 A·B·C·D·E 전부 (v1=B·C·D, v2=A만)
- **배포 중** (HF Space + Docker 둘 다 v3). 체크포인트 `retrained/best_model_v3.pt`, `train_v3.log`

> ⚠️ API/데모 입력 변경: v3는 `available`(A~E 가용 5-bool) 필드 필수.
> Gradio엔 가용성 체크박스 추가됨. 추론 = `model.recommend()` (마스킹+MOQ보정).

---

## 📋 남은 작업 (TODO)

- [x] ~~MOQ 보정 / 재학습(정책붕괴 해소) / 정책복잡도(v3)~~ → 완료
- [x] ~~FastAPI·Dockerfile·HF 전부 v3 동기화~~ → 완료
- [ ] **STAGE 3 (선택)**: Docker 이미지 레지스트리 push (Docker Hub / AWS ECR)
- [ ] **AWS EC2 배포 (선택)**: 교육계정으로 `docker run` 실습 (클라우드 교안 07 연계)
- [ ] (선택) v3 가용성 확률·페널티 튜닝, 멀티업체 동시발주(분할주문) 액션 확장

---

## 📌 메모

- 영구무료 우선순위: HF Spaces > AWS(신규계정 6개월/크레딧 제한)
- Docker 이미지 경량화: CPU torch 인덱스(`download.pytorch.org/whl/cpu`) 사용 → 이미지 1.31GB (GPU판이면 ~6GB+)
- HF Space는 push 시 자동 재빌드. 별도 서버 관리 불필요.
