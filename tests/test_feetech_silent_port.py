"""The Feetech receive path waits for replies without consuming a core."""

import os
import pty
import threading
import time

from orca_core.hardware.feetech import (
    COMM_RX_CORRUPT,
    COMM_RX_TIMEOUT,
    COMM_SUCCESS,
    PortHandler,
    sms_sts,
)


def _port():
    master, slave = pty.openpty()
    handler = PortHandler(os.ttyname(slave))
    handler.baudrate = 115200  # Portable PTY rate; reply behavior is baud-independent.
    assert handler.openPort()
    return master, slave, handler


def _close(master, slave, handler):
    handler.closePort()
    os.close(master)
    os.close(slave)


def test_unanswered_ping_waits_for_deadline_without_busy_spin():
    master, slave, handler = _port()
    try:
        start_wall = time.monotonic()
        start_cpu = time.process_time()
        _model, result, _error = sms_sts(handler).ping(1)
        wall = time.monotonic() - start_wall
        cpu = time.process_time() - start_cpu
        assert result == COMM_RX_TIMEOUT
        assert 0.04 <= wall < 0.2
        assert cpu < 0.025
    finally:
        _close(master, slave, handler)


def test_complete_reply_is_accepted_before_deadline():
    master, slave, handler = _port()
    writer = threading.Thread(target=lambda: (time.sleep(0.005), os.write(master, b"\xff\xff\x01\x02\x00\xfc")))
    try:
        handler.setPacketTimeoutMillis(50)
        writer.start()
        packet, result = sms_sts(handler).rxPacket()
        assert result == COMM_SUCCESS
        assert packet == [0xff, 0xff, 1, 2, 0, 0xfc]
    finally:
        writer.join(timeout=1)
        _close(master, slave, handler)


def test_partial_reply_expires_as_corrupt():
    master, slave, handler = _port()
    writer = threading.Thread(target=lambda: (time.sleep(0.005), os.write(master, b"\xff\xff\x01")))
    try:
        handler.setPacketTimeoutMillis(50)
        writer.start()
        _packet, result = sms_sts(handler).rxPacket()
        assert result == COMM_RX_CORRUPT
    finally:
        writer.join(timeout=1)
        _close(master, slave, handler)
