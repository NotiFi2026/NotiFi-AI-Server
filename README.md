# NotiFi AI Server

CSI 추론 모델과 LangGraph 에스컬레이션 에이전트를 함께 담당하는 FastAPI 서버.
Spring 백엔드(`NotiFi-Server`)와는 내부 API(I1~I5)로, 모델 패키지(`notifi-ai`)와는 인프로세스로 연결된다.

```
ESP32 CSI  →  [이 서버]  →  Spring 백엔드  →  보호자 앱
              추론 + 에이전트     이벤트·에스컬레이션 저장
```

---

## 셋업

**Python 3.11이 필요하다.** torch는 3.14용 휠이 없다.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
# torch CUDA 휠은 PyPI에 없어 전용 인덱스가 필요하다 (CPU만 쓸 거면 이 줄 생략)
.venv\Scripts\pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
# 모델 패키지는 형제 레포에 있다. artifacts(~109MB)가 package-data가 아니라 editable로 설치한다
.venv\Scripts\pip install -e ..\NotiFi-CSI-to-Pose\NotiFi_AI_v1
```

모델 패키지는 `NotiFi-CSI-to-Pose` 레포의 **`feature/notifi-ai-v1` 브랜치**에 있다. 이 레포로 가중치를 복사하지 않는다 — 소스 오브 트루스는 한 곳이고, AI팀 갱신은 그쪽에서 pull하면 그대로 반영된다.

`.env`는 `.env.example`을 복사해 만든다.

## 실행

```powershell
.venv\Scripts\python -m uvicorn main:app --port 8010
```

기동 시 모델을 로드하고 warmup(합성 윈도 1회 추론)까지 마친 뒤 포트가 열린다. 첫 CUDA 호출이 수십 초 걸리는 것을 기동 시점으로 옮긴 것이라 **기동은 수 초~수십 초 걸린다**. 모델 없이 에이전트만 개발하려면 `NOTIFI_MODEL_ENABLED=false`.

## 엔드포인트

| Method | Path | 인증 | 설명 |
|---|---|---|---|
| POST | `/internal/agent/run` | `X-Internal-Key` | `ModelResult` 수신 → 백그라운드로 에스컬레이션 실행, 202 |
| GET | `/internal/model/health` | 없음 | 모델 로드 상태 + 메타데이터(17행동·3위험도) |
| POST | `/internal/model/devices/{device_id}/predict` | `X-Internal-Key` | 쿼리 NPZ 추론. `?include_pose=true`면 SMPL-22 좌표 포함 |
| GET | `/status` | 없음 | 앱 폴링용 현재 위험도 (데모) |
| POST | `/status/demo` | 없음 | 위험도 수동 변경 (데모 전용) |

쿼리 NPZ는 `csi [T≤304, 3, 114, 2] float32` + `link_mask [T, 3] bool`. 30Hz 기준 304프레임 ≈ 10.13초 윈도다.

## 모델 런타임 동작

`app/model/runtime.py` — 모델 패키지의 `notifi_ai/api.py:create_app`이 하던 구성을 옮기면서 두 가지를 보강했다.

- **lifespan 로드 + warmup** (`main.py`): 원본은 warmup을 호출하지 않아 첫 요청이 수십 초 걸렸다. 로드에 실패해도 서버는 뜬다 — 에스컬레이션 에이전트는 모델 없이도 동작해야 하므로 예외를 삼키고 error 로그만 남긴다.
- **블로킹 추론을 이벤트 루프 밖으로**: 모델은 스레드 안전하지 않아 `threading.Lock`으로 직렬화하고, 라우터에서 `run_in_threadpool`로 호출한다. 따라서 **동시 추론은 1건**이다.

캘리브레이션 프로필이 없는 디바이스는 400이다. 무보정 추론은 허용하지 않는다(확정된 연동 계약). 프로필은 `NOTIFI_REGISTRY_ROOT`(기본 `runtime/devices/{device_id}/`) 아래에 있고, 디바이스 등록·캘리브레이션 API는 아직 없다(세션 5 예정).

## 검증

```powershell
# 설치 무결성 — artifacts sha256 5개 + CUDA 스모크
.venv\Scripts\python ..\NotiFi-CSI-to-Pose\NotiFi_AI_v1\scripts\verify_release.py --smoke --device cuda
# 모델 상태
curl http://127.0.0.1:8010/internal/model/health
```

## 트러블슈팅

| 증상 | 원인·해결 |
|---|---|
| `torch` 설치 실패 / 휠 없음 | venv가 Python 3.12+ 다. 3.11로 다시 만든다 |
| `NotiFi_AI_v1 artifacts are missing` | editable 설치가 아니거나 모델 레포가 다른 브랜치다. `feature/notifi-ai-v1` 체크아웃 확인 |
| predict 400 `calibration.pt` | 해당 device_id의 캘리브레이션 프로필이 없다 |
| `Form data requires "python-multipart"` | `pip install -r requirements.txt` 재실행 |
| 기동이 느리다 | 정상 — warmup이 첫 요청 지연을 기동으로 옮긴 것이다 |
