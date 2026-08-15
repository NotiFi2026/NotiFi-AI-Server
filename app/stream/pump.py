"""수집 데몬 본체 — 라인을 받아 윈도를 만들고 추론·적재까지 이어붙인다.

    소스(시리얼) ─수신 스레드─→ 라우터 ─→ 링버퍼
                                            ↓ stride마다
                                     윈도 → 추론 → ingest_service → Spring

수신은 블로킹이라 스레드에서 돌리고, 윈도·추론은 이벤트 루프의 태스크로 돈다.

`notifi_ai`를 import하므로 **부팅 경로에서 import하면 안 된다**(lifespan 안에서만).
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi.concurrency import run_in_threadpool

from app.agent.schemas import RiskLevel
from app.clients import spring_client
from app.common.logging_config import logger
from app.config import Settings
from app.model import ingest_service, pipeline
from app.model.adapter import window_from_packets
from app.model.errors import InferenceBusyError, ModelContractError
from app.model.runtime import ModelRuntime
from app.stream.buffer import BufferSet
from app.stream.collector import SessionStore
from app.stream.policy import AlertCooldown
from app.stream.router import PacketRouter
from app.stream.sources import LineSource, build_source


class StreamPump:
    """소스 하나에서 오는 CSI를 등록된 디바이스별로 갈라 실시간 판정한다."""

    def __init__(self, runtime: ModelRuntime, config: Settings) -> None:
        self._runtime = runtime
        self._config = config
        spec = runtime.spec

        # 윈도 하나를 온전히 채우려면 window_seconds가 필요하고, 처리 지연을 감안해 여유를 둔다
        self.buffers = BufferSet(spec.links, retain_seconds=spec.window_seconds * 2)
        self.router = PacketRouter(self.buffers)
        self.cooldown = AlertCooldown(config.notifi_stream_cooldown_seconds)
        # 캘리브레이션 캡처가 같은 버퍼를 본다 — 수집 경로가 하나뿐이라 어긋날 수 없다
        self.sessions = SessionStore(self.buffers, spec)

        self._source: LineSource | None = None
        self._reader: threading.Thread | None = None
        self._task: asyncio.Task | None = None
        self._stop = threading.Event()
        #: 디바이스별 마지막으로 처리한 윈도 끝 — 같은 구간을 두 번 추론하지 않는다
        self._last_end: dict[str, float] = {}
        #: 수집 시작 시각. 버퍼가 한 윈도를 채우기 전에는 판정하지 않는다
        self._started_at: float | None = None
        #: 진행 중인 에스컬레이션 에이전트 태스크. GC로 사라지지 않게 붙잡아 둔다
        self._agent_tasks: set[asyncio.Task] = set()
        #: 보드별 마지막 하트비트 전송 시각 — 주기를 지키는 기준
        self._last_heartbeat: dict[str, float] = {}
        #: 발송 중인 하트비트 태스크. 에이전트와 달리 정지 시 취소해도 된다(부가 정보다)
        self._heartbeat_tasks: set[asyncio.Task] = set()
        #: Spring에 등록되지 않아 404가 난 보드. 주기마다 같은 경고를 반복하지 않는다
        self._unregistered_warned: set[str] = set()
        self.stats = {
            "lines": 0,
            "packets": 0,
            "windows": 0,
            "sent": 0,
            "skipped_cooldown": 0,
            "heartbeats": 0,
        }

    # ── 수명주기 ────────────────────────────────────────────────────────────

    def start(self, devices: list) -> None:
        self.router.reload(devices)
        self._source = build_source(
            self._config.notifi_stream_source,
            port=self._config.notifi_stream_port,
            baud=self._config.notifi_stream_baud,
            replay_path=self._config.notifi_stream_replay_path,
            replay_speed=self._config.notifi_stream_replay_speed,
            replay_loop=self._config.notifi_stream_replay_loop,
        )
        self._stop.clear()
        self._started_at = time.monotonic()
        self._reader = threading.Thread(target=self._read_loop, name="csi-reader", daemon=True)
        self._reader.start()
        self._task = asyncio.create_task(self._window_loop())
        logger.info(
            "수집 데몬 기동",
            extra={
                "action": "stream_started",
                "source": self._config.notifi_stream_source,
                "stride_seconds": self._config.notifi_stream_stride_seconds,
                "boards": self.router.known_boards(),
            },
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._source is not None:
            self._source.close()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._reader is not None:
            # 시리얼 timeout이 0.5초라 그 안에 빠져나온다. 안 죽어도 daemon 스레드라 종료를 막지 않는다
            self._reader.join(timeout=3.0)
        for task in list(self._heartbeat_tasks):
            # 에스컬레이션과 달리 취소해도 된다 — 못 보낸 하트비트는 다음 기동이 채운다
            task.cancel()
        if self._agent_tasks:
            # 진행 중인 응급 대응은 취소하지 않는다 — 119 신고 직전에 끊는 셈이 된다.
            # 서버가 정말 내려가면 어차피 같이 죽지만, 그 사실은 남겨야 원인을 안다.
            logger.warning(
                "정지 시점에 진행 중인 에스컬레이션이 있다",
                extra={"action": "stream_stop_with_active_agents", "count": len(self._agent_tasks)},
            )
        logger.info("수집 데몬 정지", extra={"action": "stream_stopped", **self.stats})

    # ── 수신 ────────────────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        assert self._source is not None
        try:
            for line in self._source.lines():
                if self._stop.is_set():
                    break
                self.stats["lines"] += 1
                if self.router.handle(line, time.monotonic()):
                    self.stats["packets"] += 1
        except Exception as exc:  # noqa: BLE001 - 수신이 죽어도 서버는 살아 있어야 한다
            logger.error(
                "수집 수신 중단",
                extra={"action": "stream_reader_failed", "error": str(exc)},
                exc_info=True,
            )

    # ── 윈도·판정 ───────────────────────────────────────────────────────────

    async def _window_loop(self) -> None:
        stride = self._config.notifi_stream_stride_seconds
        while not self._stop.is_set():
            try:
                await asyncio.sleep(stride)
                self._flush_heartbeats(time.monotonic())
                for buffer in self.buffers.active():
                    await self._process(buffer.device_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 한 윈도 실패로 루프가 죽으면 안 된다
                logger.error(
                    "윈도 처리 실패",
                    extra={"action": "stream_window_failed", "error": str(exc)},
                    exc_info=True,
                )

    def _flush_heartbeats(self, now: float) -> None:
        """송신 중인 보드마다 Spring에 생존 신호를 보낸다(I4).

        보드가 살아 있다는 걸 아는 건 CSI 라인을 받는 여기뿐이다. 이 호출이 없으면
        Spring의 `last_seen_at`이 갱신되지 않아, 노드가 멀쩡히 도는 동안에도 보호자 앱
        디바이스 화면은 "신호 없음"으로 남는다.

        **판정 루프를 기다리게 하지 않는다 — 그래서 await가 아니라 태스크다.**
        예외를 삼키는 것만으로는 부족하다. 여기서 `await`로 보내면 Spring이 느리거나
        응답하지 않을 때 보드 수 × 타임아웃(5초)만큼 윈도 처리가 통째로 밀린다.
        낙상 판정이 20초 늦는 것과 디바이스 화면이 한 주기 늦게 갱신되는 것은
        비교 대상이 아니다.

        Spring이 죽어 있어도 실패한 보드는 주기만큼 쉬었다 재시도한다(stride마다
        재시도하면 로그가 덮인다). 임계가 5분이라 한 주기 늦어도 상태는 뒤집히지 않는다.
        """
        interval = self._config.notifi_stream_heartbeat_seconds
        if interval <= 0:
            return

        for device_uid in self.router.alive_boards(since=now - interval):
            last = self._last_heartbeat.get(device_uid)
            if last is not None and now - last < interval:
                continue
            # 보내기 전에 찍는다 — 발송이 느려도 다음 stride가 중복 발송하지 않게
            self._last_heartbeat[device_uid] = now
            # 에이전트 태스크와 같은 이유로 반환값을 붙잡는다: 루프는 약한 참조만 들고 있다
            task = asyncio.create_task(self._send_heartbeat(device_uid))
            self._heartbeat_tasks.add(task)
            task.add_done_callback(self._heartbeat_tasks.discard)

    async def _send_heartbeat(self, device_uid: str) -> None:
        """하트비트 한 건. 실패는 여기서 끝난다 — 호출부는 윈도 루프다."""
        try:
            registered = await spring_client.send_heartbeat(device_uid)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 하트비트 실패가 감지를 막으면 안 된다
            logger.warning(
                "하트비트 전송 실패",
                extra={
                    "action": "stream_heartbeat_failed",
                    "device_uid": device_uid,
                    "error": str(exc),
                },
            )
            return

        if registered:
            self.stats["heartbeats"] += 1
            self._unregistered_warned.discard(device_uid)
        elif device_uid not in self._unregistered_warned:
            # 보드는 켰는데 앱에서 D1 등록을 안 했거나 MAC 표기가 어긋난 상태다.
            # Spring은 device_uid를 완전 일치로 찾으므로 대소문자만 달라도 여기로 온다
            self._unregistered_warned.add(device_uid)
            logger.warning(
                "Spring에 등록되지 않은 보드 — 하트비트 무시됨",
                extra={"action": "stream_heartbeat_unregistered", "device_uid": device_uid},
            )

    async def _process(self, device_id: str) -> None:
        spec = self._runtime.spec
        buffer = self.buffers.get(device_id)
        end = buffer.newest_packet_at()
        if end is None:
            return

        # 아직 한 윈도를 채울 만큼 안 모였으면 기다린다.
        # 반쪽 윈도는 결측투성이라 저품질로 강등되고, 강등된 danger는 경보를 못 울린다 —
        # 기동 직후 몇 초를 판정에 쓰면 그 시간이 통째로 사각지대가 된다.
        # (monotonic()의 원점은 임의라 `end < window_seconds` 같은 비교는 성립하지 않는다.)
        if self._started_at is None or end - self._started_at < spec.window_seconds:
            return

        start = end - spec.window_seconds
        previous = self._last_end.get(device_id)
        if previous is not None and end - previous < self._config.notifi_stream_stride_seconds:
            return  # 새 데이터가 stride만큼 쌓이지 않았다

        per_link_times, per_link_iq = buffer.window(start, end)
        if not any(len(times) for times in per_link_times):
            return

        csi, link_mask = window_from_packets(per_link_times, per_link_iq, spec, end)
        coverage = [float(link_mask[:, i].mean()) for i in range(link_mask.shape[1])]
        self._last_end[device_id] = end
        self.stats["windows"] += 1

        care_target_id = pipeline.care_target_id_from(device_id)
        if care_target_id is None:
            # 규약 밖 device_id는 Spring의 어느 노인인지 알 수 없다 — 추론해도 적재할 곳이 없다
            logger.warning(
                "규약 밖 device_id — 적재 생략",
                extra={"action": "stream_unmappable_device", "device_id": device_id},
            )
            return

        try:
            pred = await run_in_threadpool(
                self._runtime.predict_arrays, device_id, csi, link_mask, True
            )
        except InferenceBusyError:
            # 앞 추론이 밀렸다. 재시도하지 않고 이 윈도를 버린다 — stride 뒤에 새 윈도가 온다
            logger.warning(
                "추론 포화 — 윈도 버림",
                extra={"action": "stream_window_dropped", "device_id": device_id},
            )
            return
        except FileNotFoundError:
            # 캘리브레이션 전이다. 매 stride마다 시끄럽지 않게 info로 남긴다
            logger.info(
                "캘리브레이션 없음 — 추론 생략",
                extra={"action": "stream_needs_calibration", "device_id": device_id},
            )
            return

        await self._deliver(device_id, care_target_id, pred, end, coverage)

    async def _deliver(
        self,
        device_id: str,
        care_target_id: int,
        pred: dict[str, Any],
        window_end: float,
        coverage: list[float],
    ) -> None:
        """판정 결과를 Spring으로. 위험 판정은 쿨다운으로 중복 신고를 막는다.

        **억제 판단은 강등이 끝난 뒤에 한다.** 모델 원본 라벨로 판단하면, 저품질이라 WARNING으로
        내려가 에스컬레이션을 만들지도 않은 윈도가 쿨다운을 걸어버린다. 그 사이에 일어난
        진짜 낙상은 억제되어 아무에게도 알려지지 않는다.
        """
        # 버퍼 시각은 monotonic이라 벽시계로 되돌린다. 처리 직후라 보정폭은 수십 ms다
        detected_at = datetime.now(timezone.utc) - timedelta(
            seconds=max(0.0, time.monotonic() - window_end)
        )

        def gate(model_result: Any) -> bool:
            """Spring은 DANGER 이벤트마다 에스컬레이션을 새로 만든다(중복 판정 없음).
            겹치는 윈도를 그대로 보내면 한 번 넘어졌는데 119 신고가 여러 번 걸린다."""
            if model_result.risk_level is not RiskLevel.DANGER:
                return True
            if self.cooldown.allows(device_id, window_end):
                return True
            self.stats["skipped_cooldown"] += 1
            logger.info(
                "위험 재신고 억제",
                extra={
                    "action": "stream_alert_cooldown",
                    "device_id": device_id,
                    "remaining_seconds": round(self.cooldown.remaining(device_id, window_end), 1),
                },
            )
            return False

        try:
            result = await ingest_service.deliver(
                pred,
                device_id=device_id,
                care_target_id=care_target_id,
                spring_device_id=None,
                detected_at=detected_at,
                schedule_danger=self._schedule_danger,
                gate=gate,
            )
        except ModelContractError as exc:
            logger.error(
                "모델 계약 위반",
                extra={"action": "model_contract_error", "device_id": device_id, "error": str(exc)},
            )
            return
        except httpx.HTTPError as exc:
            # 적재 실패는 재시도하지 않는다 — 10초 뒤 새 윈도가 온다.
            # mark_sent도 안 됐으므로 다음 NORMAL이 절감으로 막히지 않는다
            logger.error(
                "Spring 적재 실패 — 윈도 버림",
                extra={"action": "spring_ingest_failed", "device_id": device_id, "error": str(exc)},
            )
            return

        if result.get("sent"):
            self.stats["sent"] += 1
            if result.get("escalation_id"):
                # **대응이 붙었으면** 억제한다 — 새로 시작했든(triggered) 진행 중인 건에
                # 합쳐졌든(reused) 이 낙상은 이미 처리되고 있다.
                #
                # "모델이 danger라 했다"나 "적재됐다"로 판단하면 안 되고, 반대로
                # escalation_triggered만 보면 재사용된 윈도가 억제를 못 걸어 stride(2초)마다
                # I1이 계속 나간다. Spring이 대응 id를 주는 것이 유일하게 정직한 신호다.
                self.cooldown.mark(device_id, window_end)
            logger.info(
                "실시간 판정 적재",
                extra={
                    "action": "stream_ingested",
                    "device_id": device_id,
                    "activity_class": result.get("activity_class"),
                    "risk_level": result.get("risk_level"),
                    "escalation_triggered": result.get("escalation_triggered"),
                    "link_coverage": [round(value, 3) for value in coverage],
                },
            )

    def _schedule_danger(self, model_result: Any, saved: dict[str, Any]) -> None:
        """에이전트는 수 분짜리 흐름이라 윈도 루프를 붙잡으면 안 된다.

        **반환된 태스크를 반드시 붙잡아 둔다.** 이벤트 루프는 태스크를 약한 참조로만 들고 있어서,
        참조를 버리면 실행 도중 GC 대상이 된다. 에스컬레이션은 음성확인 → 60초 대기 → 보호자
        알림 → 119로 이어지는 흐름이라, 사라지면 **119 신고 직전에 조용히 멈춘다.**
        """
        from app.api.routes import run_agent_safely

        task = asyncio.create_task(run_agent_safely(model_result, saved))
        self._agent_tasks.add(task)
        task.add_done_callback(self._agent_tasks.discard)
