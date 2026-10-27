"""Serial failures must leave the bus available for disable/recovery traffic."""
import math

import pytest
import serial

from orca_core.hardware.feetech import port_handler as port_module
from orca_core.hardware.feetech.protocol_packet_handler import protocol_packet_handler
from orca_core.hardware.feetech.scservo_def import COMM_RX_FAIL, COMM_SUCCESS, COMM_TX_FAIL


class Serial:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.write_error = self.read_error = self.clear_error = None
        self.clears = 0
        self.writes = []

    def reset_input_buffer(self):
        if self.clear_error:
            raise self.clear_error
        self.clears += 1

    def flush(self):
        pytest.fail("unbounded output drain cannot be used to discard stale input")

    def write(self, packet):
        self.writes.append((tuple(packet), self.write_timeout))
        if self.write_error:
            raise self.write_error
        return len(packet)

    def read(self, length):
        if self.read_error:
            raise self.read_error
        return b""

    def close(self):
        pass


@pytest.fixture
def bus(monkeypatch):
    monkeypatch.setattr(port_module.serial, "Serial", Serial)
    port = port_module.PortHandler("test-boundary")
    assert port.openPort()
    return port, protocol_packet_handler(port, 0)


def packet():
    return [0, 0, 1, 2, 1, 0]


@pytest.mark.parametrize("baud", [4800, 115200, 1000000])
@pytest.mark.parametrize("size", [6, 250])
def test_writes_use_wire_time_plus_existing_turnaround_and_latency_budget(bus, baud, size):
    port, _ = bus
    assert port.setBaudRate(baud)
    assert math.isfinite(port.ser.write_timeout) and port.ser.write_timeout > 0
    port.writePort(bytes(size))
    expected = (size + 3) * 10 / baud + 0.050
    assert port.ser.write_timeout == pytest.approx(expected)
    port.setPacketTimeout(size)
    assert port.packet_timeout / 1000 == pytest.approx(expected)


@pytest.mark.parametrize("failure", [serial.SerialTimeoutException("write timed out"), serial.SerialException("port lost"), OSError("disconnected")])
@pytest.mark.parametrize("step", ["write_error", "clear_error"])
def test_failed_transmit_releases_owner_and_allows_next_disable_packet(bus, failure, step):
    port, protocol = bus
    setattr(port.ser, step, failure)
    assert protocol.txPacket(packet()) == COMM_TX_FAIL
    assert not port.is_using
    setattr(port.ser, step, None)
    assert protocol.txPacket(packet()) == COMM_SUCCESS
    assert port.is_using  # Retained until the response is consumed.
    assert port.ser.clears >= 2


@pytest.mark.parametrize("failure", [serial.SerialException("read failed"), OSError("port removed")])
def test_failed_receive_releases_owner_and_allows_cleanup_transmit(bus, failure):
    port, protocol = bus
    assert protocol.txPacket(packet()) == COMM_SUCCESS
    port.ser.read_error = failure
    received, result = protocol.rxPacket()
    assert received == [] and result == COMM_RX_FAIL
    assert not port.is_using
    port.ser.read_error = None
    assert protocol.txPacket(packet()) == COMM_SUCCESS


@pytest.mark.parametrize("failure", [serial.SerialException("aggregate read failed"), OSError("port removed")])
def test_failed_sync_receive_releases_owner_for_disable(bus, failure):
    port, protocol = bus
    port.is_using = True
    port.ser.read_error = failure
    result, received = protocol.syncReadRx(4, 2)
    assert result == COMM_RX_FAIL and received == []
    assert not port.is_using
    port.ser.read_error = None
    assert protocol.txPacket(packet()) == COMM_SUCCESS
