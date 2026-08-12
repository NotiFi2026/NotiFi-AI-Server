"""라인 → (디바이스, 링크) 배정 → 버퍼 적재.

MAC이 링크를 정한다. 매핑은 이미 레지스트리에 있다 — `register_device`가 받는
`tx1_id`/`tx2_id`/`tx3_id`가 그대로 링크 0/1/2다. 별도 설정 파일을 만들지 않는다.

`notifi_ai`를 import하므로 **부팅 경로에서 import하면 안 된다**(lifespan 안에서만).
"""
from __future__ import annotations

import threading

from app.common.logging_config import logger
from app.model.adapter import parse_csi_line
from app.stream.buffer import BufferSet


def _normalize(board_id: str) -> str:
    """MAC 표기 흔들림(대문자·공백)을 흡수한다 — 등록 오타가 조용한 무신호가 되지 않게."""
    return board_id.strip().lower()


class PacketRouter:
    """등록된 보드에서 온 패킷만 버퍼에 넣는다.

    모르는 MAC은 버린다. 옆집 노드나 시험용 보드가 남의 가구 신호에 섞이면
    낙상 판정이 조용히 오염되기 때문이다. 대신 처음 본 MAC은 한 번씩 로그로 남겨
    "보드는 켰는데 등록을 안 했다"를 알아챌 수 있게 한다.
    """

    #: TX 링크 순서. DeviceConfig 필드명과 1:1이며 인덱스가 곧 링크 번호다
    LINK_FIELDS = ("tx1_id", "tx2_id", "tx3_id")

    def __init__(self, buffers: BufferSet) -> None:
        self._buffers = buffers
        self._lock = threading.Lock()
        self._by_mac: dict[str, tuple[str, int]] = {}
        self._unknown_seen: set[str] = set()

    def reload(self, devices: list) -> None:
        """레지스트리의 DeviceConfig 목록으로 매핑을 다시 만든다(등록·삭제 후 호출)."""
        mapping: dict[str, tuple[str, int]] = {}
        for config in devices:
            for index, field in enumerate(self.LINK_FIELDS):
                board_id = getattr(config, field, None)
                if board_id:
                    mapping[_normalize(board_id)] = (config.device_id, index)
        with self._lock:
            self._by_mac = mapping
            self._unknown_seen.clear()
        logger.info(
            "수집 매핑 갱신",
            extra={"action": "stream_map_reload", "boards": len(mapping)},
        )

    def handle(self, line: str, at: float) -> bool:
        """라인 한 줄을 처리한다. 버퍼에 넣었으면 True."""
        parsed = parse_csi_line(line)
        if parsed is None:
            return False
        sender, iq = parsed
        mac = _normalize(sender)

        with self._lock:
            target = self._by_mac.get(mac)
            first_time = target is None and mac not in self._unknown_seen
            if target is None:
                self._unknown_seen.add(mac)

        if target is None:
            if first_time:
                # 등록 안 된 보드가 켜져 있다는 신호 — 설치 현장에서 제일 흔한 실수다
                logger.warning(
                    "등록되지 않은 보드의 패킷",
                    extra={"action": "stream_unknown_board", "sender": mac},
                )
            return False

        device_id, link_index = target
        self._buffers.get(device_id).add(link_index, at, iq)
        return True

    def known_boards(self) -> int:
        with self._lock:
            return len(self._by_mac)

    def unknown_boards(self) -> list[str]:
        with self._lock:
            return sorted(self._unknown_seen)
