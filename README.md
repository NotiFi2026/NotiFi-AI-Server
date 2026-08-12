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
| GET | `/internal/model/health` | 선택 | 모델 로드 상태(17행동·3위험도). 키를 주면 설치 경로·성능지표·등록 디바이스까지 |
| POST | `/internal/model/devices/{device_id}/predict` | `X-Internal-Key` | 쿼리 NPZ 추론만 하고 결과를 돌려준다(순수 프로브). `?include_pose=true`면 SMPL-22 좌표 포함 |
| POST | `/internal/model/devices/{device_id}/ingest` | `X-Internal-Key` | **추론 → Spring 적재 → 에스컬레이션** 전체 파이프라인, 202 |
| GET | `/status` | 없음 | 앱 폴링용 현재 위험도 (데모) |
| POST | `/status/demo` | 없음 | 위험도 수동 변경 (데모 전용) |

쿼리 NPZ는 `csi [T≤304, 3, 114, 2] float32` + `link_mask [T, 3] bool`. 30Hz 기준 304프레임 ≈ 10.13초 윈도다. 업로드 상한 8MB, `device_id`는 `[A-Za-z0-9_-]{1,64}`만 허용한다(레지스트리 경로 세그먼트로 쓰이므로).

### ingest 파이프라인

`ingest`는 CSI 윈도 하나를 받아 보호자 알림까지 이어지는 경로 전체를 담당한다.

```
NPZ → 추론 → ModelResult 변환 → (NORMAL 절감 판단) → I1 적재
                                                    → 비정상이면 I5 클립
                                                    → danger면 에스컬레이션(백그라운드)
```

폼 필드: `file`(NPZ), `care_target_id`(필수), `spring_device_id`(선택), `window_end_at`(선택, 기본 now).

- **`care_target_id`를 호출자가 준다** — 모델 레지스트리의 문자열 `device_id`와 Spring의 노인 ID를 잇는 수단이 아직 없다(세션 5에서 정식화).
- **`window_end_at`도 호출자가 준다** — 모델은 시각을 모른다. 윈도 시작은 `프레임수/fps`로 역산한다.
- **NORMAL 절감**: 10초 윈도를 상시 추론하면 NORMAL이 폭증하므로, 행동이 바뀌면 즉시 보내고 같은 행동이 이어지면 `NOTIFI_NORMAL_INTERVAL_SECONDS`(기본 300초)에 1건만 보낸다. 비정상 이벤트는 절대 거르지 않는다. 상태는 **인메모리**라 재시작하면 초기화된다(NORMAL 1건을 더 보내는 정도의 영향).
- **저품질 강등**: `quality.low_quality`면 danger 판정이라도 WARNING으로 낮추고 원 판정을 `features`에 남긴다 — 링크 부족만으로 자동 경보를 울리지 않는다.
- **I5는 I1보다 먼저 기다리지 않는다**: danger 흐름은 음성확인·대기로 수 분이 걸리므로, 파이프라인이 I1·I5를 **동기로 먼저** 끝내고 에이전트에는 받은 ID를 넘겨 재전송을 막는다. 보호자가 알림을 받는 시점에 리플레이가 이미 있다.
- Spring 적재 실패는 삼키지 않고 **502**로 올린다 — 호출자가 재시도해야 한다.

`SPRING_INTERNAL_KEY`를 설정하지 않으면 내부 API는 **모든 요청을 401로 거부한다.** 빈 키를 유효한 키로 취급하면 헤더 없는 요청이 통과해 전부 무인증으로 열리기 때문이다.

응답 코드: 입력 오류 400 / 인증 401 / 업로드 초과 413 / 모델 미로드 503 / 추론 자체 실패 500. **GPU 장애를 400으로 내리지 않는다** — 클라이언트가 재시도할 수 있어야 한다.

## 모델 런타임 동작

`app/model/runtime.py` — 모델 패키지의 `notifi_ai/api.py:create_app`이 하던 구성을 옮기면서 두 가지를 보강했다.

- **lifespan 로드 + warmup** (`main.py`): 원본은 warmup을 호출하지 않아 첫 요청이 수십 초 걸렸다. 로드에 실패해도 서버는 뜬다 — 에스컬레이션 에이전트는 모델 없이도 동작해야 하므로 예외를 삼키고 error 로그만 남긴다. 이때 `/internal/model/health`는 503을 반환하므로 모니터링이 "모델 없는 서버"를 정상으로 오인하지 않는다.

`notifi_ai`(및 torch)는 **lifespan 안에서만 import한다.** 라우터가 타입 힌트 때문에 top-level로 import하면 모델 미설치 환경에서 서버 자체가 부팅되지 않는다 — 실제로 났던 회귀라 `tests/test_model_api.py::test_boots_without_notifi_ai`가 가드한다.
- **블로킹 추론을 이벤트 루프 밖으로**: 모델은 스레드 안전하지 않아 `threading.Lock`으로 직렬화하고, 라우터에서 `run_in_threadpool`로 호출한다. 따라서 **동시 추론은 1건**이다.

캘리브레이션 프로필이 없는 디바이스는 400이다. 무보정 추론은 허용하지 않는다(확정된 연동 계약). 프로필은 `NOTIFI_REGISTRY_ROOT`(기본 `runtime/devices/{device_id}/`) 아래에 있고, 디바이스 등록·캘리브레이션 API는 아직 없다(세션 5 예정).

## 검증

```powershell
# 테스트 — GPU·모델 없이 돈다 (부팅·인증·입력 검증 회귀 가드)
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest tests -q
# 설치 무결성 — artifacts sha256 5개 + CUDA 스모크
.venv\Scripts\python ..\NotiFi-CSI-to-Pose\NotiFi_AI_v1\scripts\verify_release.py --smoke --device cuda
# 모델 상태
curl http://127.0.0.1:8010/internal/model/health
```

## 트러블슈팅

| 증상 | 원인·해결 |
|---|---|
| `torch` 설치 실패 / 휠 없음 | venv가 Python 3.12+ 다. 3.11로 다시 만든다 |
| 서버는 뜨는데 추론이 503 | torch·notifi-ai 미설치. `requirements.txt`만으로는 안 깔린다 — 위 셋업의 나머지 2줄을 실행한다 |
| `NotiFi_AI_v1 artifacts are missing` | editable 설치가 아니거나 모델 레포가 다른 브랜치다. `feature/notifi-ai-v1` 체크아웃 확인 |
| predict 400 `calibration.pt` | 해당 device_id의 캘리브레이션 프로필이 없다 |
| `Form data requires "python-multipart"` | `pip install -r requirements.txt` 재실행 |
| 기동이 느리다 | 정상 — warmup이 첫 요청 지연을 기동으로 옮긴 것이다 |
