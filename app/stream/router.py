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

    #: RX 보드. 라우팅에는 안 쓰이지만 생존 보고에는 필요하다(아래 alive_boards 참고)
    RX_FIELD = "rx_id"

    #: 미등록 MAC 기록 상한. 옆집 보드가 계속 잡히면 무한히 쌓여 장시간 운용에서 메모리가 샌다.
    #: 목적은 "등록 안 된 보드가 있다"를 알리는 것이라 몇 개만 남으면 충분하다.
    MAX_UNKNOWN_TRACKED = 64

    def __init__(self, buffers: BufferSet) -> None:
        self._buffers = buffers
        self._lock = threading.Lock()
        self._by_mac: dict[str, tuple[str, int]] = {}
        self._unknown_seen: set[str] = set()
        #: 정규화 MAC → 등록 당시 원문. 하트비트는 Spring에 등록된 문자열 그대로 보내야 한다
        self._registered_uid: dict[str, str] = {}
        #: 정규화 MAC → 마지막으로 패킷을 받은 시각(monotonic). 하트비트 대상 판정에 쓴다
        self._last_seen: dict[str, float] = {}
        #: device_id → RX 보드 uid. RX는 송신을 안 해 MAC 매핑에 들어갈 수 없다
        self._rx_uid: dict[str, str] = {}

    def reload(self, devices: list) -> None:
        """레지스트리의 DeviceConfig 목록으로 매핑을 다시 만든다(등록·삭제 후 호출)."""
        mapping: dict[str, tuple[str, int]] = {}
        registered: dict[str, str] = {}
        rx_uid: dict[str, str] = {}
        for config in devices:
            for index, field in enumerate(self.LINK_FIELDS):
                board_id = getattr(config, field, None)
                if board_id:
                    mac = _normalize(board_id)
                    mapping[mac] = (config.device_id, index)
                    registered[mac] = board_id
            rx_board = getattr(config, self.RX_FIELD, None)
            if rx_board:
                rx_uid[config.device_id] = rx_board
        with self._lock:
            self._by_mac = mapping
            self._registered_uid = registered
            self._rx_uid = rx_uid
            self._unknown_seen.clear()
            # 등록이 빠진 보드의 수신 기록은 버린다 — 남기면 지운 보드에 하트비트가 계속 나간다
            self._last_seen = {
                mac: at for mac, at in self._last_seen.items() if mac in mapping
            }
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
            if target is None and len(self._unknown_seen) < self.MAX_UNKNOWN_TRACKED:
                self._unknown_seen.add(mac)
            if target is not None:
                # 등록된 보드만 기록한다 — 미등록 MAC은 Spring에도 없어 하트비트를 보낼 곳이 없다
                self._last_seen[mac] = at

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

    def alive_boards(self, since: float) -> dict[str, float]:
        """`since` 이후에 살아 있는 것으로 확인된 보드의 {등록 원문 uid: 확인 시각}.

        **RX도 포함한다.** RX는 수신 전용이라 CSI 라인의 sender로 절대 등장하지 않는다.
        그 보드만 빼고 보고하면 앱 디바이스 목록에서 RX 노드 하나가 영원히 "신호 없음"으로
        남는다 — 정작 그 보드가 지금 라인을 뽑아내고 있는데도. TX 패킷이 도착했다는 사실
        자체가 그것을 받아 넘긴 RX가 살아 있다는 증거다.

        수신은 리더 스레드가, 소비는 이벤트 루프가 한다. 스냅샷을 복사해 넘기는 건
        호출부가 락 밖에서 네트워크 호출을 하기 때문이다.
        """
        alive: dict[str, float] = {}
        with self._lock:
            for mac, at in self._last_seen.items():
                if at < since:
                    continue
                uid = self._registered_uid.get(mac)
                if uid is None:
                    continue
                alive[uid] = at

                target = self._by_mac.get(mac)
                rx = self._rx_uid.get(target[0]) if target else None
                if rx is not None:
                    alive[rx] = max(alive.get(rx, at), at)
        return alive

    def known_boards(self) -> int:
        with self._lock:
            return len(self._by_mac)

    def unknown_boards(self) -> list[str]:
        with self._lock:
            return sorted(self._unknown_seen)
