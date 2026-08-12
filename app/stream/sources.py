"""CSI 라인 공급원 — 데몬이 신호를 어디서 받는지만 다르고 나머지는 같다.

지금은 시리얼(RX 보드 USB 1대). ESP가 무선으로 쏘게 되면 UdpSource를 하나 더 넣고
설정만 바꾸면 된다 — 버퍼·윈도·판정은 손대지 않는다.

`notifi_ai`를 import하지 않는다. 라인은 문자열 그대로 넘기고 파싱은 router가 한다.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterator, Protocol

from app.common.logging_config import logger


class LineSource(Protocol):
    """블로킹 제너레이터. close()가 불리면 루프를 빠져나온다."""

    def lines(self) -> Iterator[str]: ...

    def close(self) -> None: ...


class SerialSource:
    """RX 보드 시리얼. **USB는 RX 1대만 꽂으면 된다** — TX 3대는 전원만 있으면 되고,
    각 라인의 sender MAC으로 구분된다."""

    def __init__(self, port: str, baud: int) -> None:
        self._port = port
        self._baud = baud
        self._closed = False
        self._serial = None

    def lines(self) -> Iterator[str]:
        import serial  # 지연 import — 시리얼 없는 배포에서 부팅을 막지 않는다

        self._serial = serial.Serial(self._port, self._baud, timeout=0.5)
        logger.info(
            "시리얼 수집 시작",
            extra={"action": "stream_serial_open", "port": self._port, "baud": self._baud},
        )
        try:
            while not self._closed:
                raw = self._serial.readline()
                if not raw:
                    continue  # timeout — 종료 플래그를 다시 본다
                yield raw.decode(errors="ignore").strip()
        finally:
            self._serial.close()
            logger.info("시리얼 수집 종료", extra={"action": "stream_serial_close"})

    def close(self) -> None:
        self._closed = True


class ReplaySource:
    """수집 CSV를 실시간처럼 재생한다. **하드웨어 없이 데몬 전 구간을 검증하는 수단이다.**

    학습 데이터셋의 낙상 trial을 재생하면 danger 분기까지 보드 없이 돌려볼 수 있다.
    CSV 형식은 수집 스크립트(save_csi_raw.py)가 쓰는 그대로 — `raw_line` 컬럼이 시리얼 한 줄이다.
    """

    def __init__(self, path: Path | str, speed: float = 1.0, loop: bool = False) -> None:
        self._path = Path(path)
        self._speed = max(speed, 0.01)
        self._loop = loop
        self._closed = False

    def lines(self) -> Iterator[str]:
        logger.info(
            "리플레이 수집 시작",
            extra={"action": "stream_replay_open", "path": str(self._path), "speed": self._speed},
        )
        while not self._closed:
            yield from self._play_once()
            if not self._loop:
                break

    def _play_once(self) -> Iterator[str]:
        with self._path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            base_ms: float | None = None
            started = time.monotonic()
            for row in reader:
                if self._closed:
                    return
                line = row.get("raw_line") or ""
                if not line.startswith("CSI_DATA"):
                    continue
                stamp = row.get("pc_time_ms")
                if stamp is None:
                    yield line
                    continue
                offset_ms = float(stamp)
                if base_ms is None:
                    base_ms = offset_ms
                # 원래 간격대로 흘려보낸다 — 몰아서 주면 링크 커버리지가 실제와 달라진다
                due = (offset_ms - base_ms) / 1000.0 / self._speed
                delay = due - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(min(delay, 1.0))
                yield line

    def close(self) -> None:
        self._closed = True


def build_source(kind: str, *, port: str, baud: int, replay_path: str, replay_speed: float,
                 replay_loop: bool) -> LineSource:
    if kind == "serial":
        return SerialSource(port, baud)
    if kind == "replay":
        if not replay_path:
            raise ValueError("replay 소스에는 NOTIFI_STREAM_REPLAY_PATH가 필요하다")
        return ReplaySource(replay_path, replay_speed, replay_loop)
    raise ValueError(f"알 수 없는 수집 소스: {kind}")
