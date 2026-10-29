from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import hardware
from hardware import GpuManager


@pytest.fixture
def fans(monkeypatch):
    nvml = hardware.pynvml
    mocks = {}
    for name in (
        "nvmlInit", "nvmlShutdown", "nvmlDeviceGetCount",
        "nvmlDeviceGetHandleByIndex", "nvmlDeviceGetPciInfo",
        "nvmlDeviceGetNumFans", "nvmlDeviceGetMinMaxFanSpeed",
        "nvmlDeviceGetTemperature", "nvmlDeviceGetFanSpeed_v2",
        "nvmlDeviceGetFanControlPolicy_v2", "nvmlDeviceSetFanControlPolicy",
        "nvmlDeviceSetFanSpeed_v2", "nvmlDeviceSetDefaultFanSpeed_v2",
    ):
        mocks[name] = Mock()
        monkeypatch.setattr(nvml, name, mocks[name])
    mocks["nvmlDeviceGetCount"].return_value = 1
    mocks["nvmlDeviceGetHandleByIndex"].return_value = "gpu0"
    mocks["nvmlDeviceGetPciInfo"].return_value = SimpleNamespace(busId="00000000:41:00.0")
    mocks["nvmlDeviceGetNumFans"].return_value = 2
    mocks["nvmlDeviceGetMinMaxFanSpeed"].return_value = (30, 100)
    mocks["nvmlDeviceGetTemperature"].return_value = 80
    mocks["nvmlDeviceGetFanSpeed_v2"].return_value = 80
    mocks["nvmlDeviceGetFanControlPolicy_v2"].return_value = nvml.NVML_FAN_POLICY_MANUAL
    # Ticks are advanced explicitly; no background hardware access in tests.
    monkeypatch.setattr(hardware.threading, "Thread", Mock())
    clock = [0.0]
    monkeypatch.setattr(hardware.time, "monotonic", lambda: clock[0])
    manager = GpuManager()

    def tick(temperature, seconds):
        clock[0] = seconds
        mocks["nvmlDeviceGetTemperature"].return_value = temperature
        manager._update_auto_fans()

    yield manager, mocks, tick
    manager.shutdown()


def test_auto_holds_at_79_and_requires_sustained_cooling(fans):
    manager, nvml, tick = fans
    assert manager.set_fan_auto(0)["fan_mode"] == "auto"
    writes = nvml["nvmlDeviceSetFanSpeed_v2"]
    assert writes.call_args_list == [call("gpu0", 0, 100), call("gpu0", 1, 100)]
    writes.reset_mock()
    for seconds in range(5, 65, 5):
        tick(79 if seconds % 10 else 80, seconds)
    writes.assert_not_called()
    tick(75, 65)
    tick(75, 94)
    writes.assert_not_called()
    tick(75, 95)
    assert writes.call_args_list == [call("gpu0", 0, 90), call("gpu0", 1, 90)]


def test_cooling_timer_resets_and_heating_bypasses_delay(fans):
    manager, nvml, tick = fans
    nvml["nvmlDeviceGetTemperature"].return_value = 75
    manager.set_fan_auto(0)
    writes = nvml["nvmlDeviceSetFanSpeed_v2"]
    writes.reset_mock()
    tick(70, 5)
    tick(71, 30)
    tick(70, 35)
    tick(70, 60)
    writes.assert_not_called()
    tick(80, 65)
    assert writes.call_args_list == [call("gpu0", 0, 100), call("gpu0", 1, 100)]


def test_cooling_preserves_each_band_hysteresis(fans):
    manager, nvml, tick = fans
    nvml["nvmlDeviceGetTemperature"].return_value = 65
    manager.set_fan_auto(0)
    tick(59, 5)
    tick(59, 35)
    assert nvml["nvmlDeviceSetFanSpeed_v2"].call_args == call("gpu0", 1, 60)
    tick(45, 40)
    tick(45, 70)
    assert nvml["nvmlDeviceSetFanSpeed_v2"].call_args == call("gpu0", 1, 40)


def test_brief_temperature_dip_cannot_skip_cooling_bands(fans):
    manager, nvml, tick = fans
    manager.set_fan_auto(0)
    tick(75, 5)
    tick(75, 30)
    tick(55, 35)
    writes = nvml["nvmlDeviceSetFanSpeed_v2"]
    assert writes.call_args == call("gpu0", 1, 90)
    writes.reset_mock()
    tick(55, 40)
    tick(55, 69)
    writes.assert_not_called()
    tick(55, 70)
    assert writes.call_args_list == [call("gpu0", 0, 75), call("gpu0", 1, 75)]


@pytest.mark.parametrize(("temperature", "speed"), [
    (0, 40), (49, 40), (50, 60), (64, 60), (65, 75),
    (74, 75), (75, 90), (79, 90), (80, 100), (95, 100),
])
def test_auto_curve_boundaries(fans, temperature, speed):
    manager, nvml, tick = fans
    nvml["nvmlDeviceGetTemperature"].return_value = temperature
    manager.set_fan_auto(0)
    assert nvml["nvmlDeviceSetFanSpeed_v2"].call_args_list == [
        call("gpu0", 0, speed), call("gpu0", 1, speed),
    ]


@pytest.mark.parametrize("failure", ["temperature", "write", "initial_auto", "manual"])
def test_failed_driver_restore_retries_until_recovery(fans, failure):
    manager, nvml, tick = fans
    error = hardware.pynvml.NVMLError(hardware.pynvml.NVML_ERROR_UNKNOWN)
    nvml["nvmlDeviceGetTemperature"].return_value = 40
    if failure != "initial_auto":
        manager.set_fan_auto(0)
    nvml["nvmlDeviceSetDefaultFanSpeed_v2"].side_effect = error
    nvml["nvmlDeviceSetFanControlPolicy"].side_effect = error
    if failure == "temperature":
        nvml["nvmlDeviceGetTemperature"].side_effect = error
        tick(40, 5)
    elif failure == "write":
        tick(95, 5)
    else:
        with pytest.raises(ValueError, match="Failed to set fan speed"):
            if failure == "initial_auto":
                manager.set_fan_auto(0)
            else:
                manager.set_fan_speed(0, 70)
    assert manager._fan_thread is not None
    assert not manager._auto_fans
    resets = nvml["nvmlDeviceSetDefaultFanSpeed_v2"]
    resets.reset_mock()
    tick(95, 10)
    assert resets.call_args_list == [call("gpu0", 0), call("gpu0", 1)]
    resets.reset_mock()
    resets.side_effect = None
    nvml["nvmlDeviceSetFanControlPolicy"].side_effect = None
    tick(95, 15)
    assert resets.call_args_list == [call("gpu0", 0), call("gpu0", 1)]
    resets.reset_mock()
    tick(95, 20)
    resets.assert_not_called()


@pytest.mark.parametrize("mode", ["manual", "auto"])
def test_successful_mode_change_cancels_pending_restore(fans, mode):
    manager, nvml, tick = fans
    manager.set_fan_auto(0)
    error = hardware.pynvml.NVMLError(hardware.pynvml.NVML_ERROR_UNKNOWN)
    nvml["nvmlDeviceGetTemperature"].side_effect = error
    nvml["nvmlDeviceSetDefaultFanSpeed_v2"].side_effect = error
    nvml["nvmlDeviceSetFanControlPolicy"].side_effect = error
    tick(80, 5)
    nvml["nvmlDeviceGetTemperature"].side_effect = None
    nvml["nvmlDeviceSetDefaultFanSpeed_v2"].side_effect = None
    nvml["nvmlDeviceSetFanControlPolicy"].side_effect = None
    if mode == "manual":
        manager.set_fan_speed(0, 70)
    else:
        manager.set_fan_auto(0)
    nvml["nvmlDeviceSetDefaultFanSpeed_v2"].reset_mock()
    tick(80, 10)
    nvml["nvmlDeviceSetDefaultFanSpeed_v2"].assert_not_called()


@pytest.mark.parametrize("failed_call", [
    "nvmlDeviceSetDefaultFanSpeed_v2",
    "nvmlDeviceSetFanControlPolicy",
    "nvmlDeviceGetHandleByIndex",
])
def test_restore_waits_for_all_fans_and_operations_to_recover(fans, failed_call):
    manager, nvml, tick = fans
    manager.set_fan_auto(0)
    error = hardware.pynvml.NVMLError(hardware.pynvml.NVML_ERROR_UNKNOWN)
    nvml["nvmlDeviceGetTemperature"].side_effect = error

    def fail_second_fan(handle, fan, *args):
        if fan == 1:
            raise error

    if failed_call == "nvmlDeviceGetHandleByIndex":
        nvml[failed_call].side_effect = error
    else:
        nvml[failed_call].side_effect = fail_second_fan
    tick(80, 5)
    nvml[failed_call].reset_mock()
    tick(80, 10)
    assert nvml[failed_call].called
    assert manager._pending_fan_restores == {0}
    nvml[failed_call].side_effect = None
    tick(80, 15)
    assert not manager._pending_fan_restores
    nvml[failed_call].reset_mock()
    tick(80, 20)
    nvml[failed_call].assert_not_called()


def test_manual_disables_temperature_control(fans):
    manager, nvml, tick = fans
    manager.set_fan_auto(0)
    assert manager.set_fan_speed(0, 70)["fan_mode"] == "manual"
    nvml["nvmlDeviceSetFanSpeed_v2"].reset_mock()
    tick(95, 5)
    nvml["nvmlDeviceSetFanSpeed_v2"].assert_not_called()


def test_auto_clamps_to_supported_range(fans):
    manager, nvml, tick = fans
    nvml["nvmlDeviceGetMinMaxFanSpeed"].return_value = (50, 90)
    nvml["nvmlDeviceGetTemperature"].return_value = 40
    manager.set_fan_auto(0)
    assert nvml["nvmlDeviceSetFanSpeed_v2"].call_args == call("gpu0", 1, 50)
    tick(95, 5)
    assert nvml["nvmlDeviceSetFanSpeed_v2"].call_args == call("gpu0", 1, 90)


@pytest.mark.parametrize("failure", ["temperature", "write"])
def test_auto_failure_restores_all_fans(fans, failure):
    manager, nvml, tick = fans
    nvml["nvmlDeviceGetTemperature"].return_value = 75
    manager.set_fan_auto(0)
    name = "nvmlDeviceGetTemperature" if failure == "temperature" else "nvmlDeviceSetFanSpeed_v2"
    nvml[name].side_effect = hardware.pynvml.NVMLError(hardware.pynvml.NVML_ERROR_UNKNOWN)
    tick(95, 5)
    assert not manager._auto_fans
    assert nvml["nvmlDeviceSetDefaultFanSpeed_v2"].call_args_list == [
        call("gpu0", 0), call("gpu0", 1),
    ]


def test_invalid_manual_speed_preserves_auto(fans):
    manager, nvml, tick = fans
    manager.set_fan_auto(0)
    with pytest.raises(ValueError, match="out of range"):
        manager.set_fan_speed(0, 101)
    assert manager.get_fan_status(0)["fan_mode"] == "auto"
    tick(95, 5)
    assert nvml["nvmlDeviceSetFanSpeed_v2"].call_args == call("gpu0", 1, 100)


def test_initial_auto_write_failure_rolls_back_all_fans(fans):
    manager, nvml, tick = fans
    nvml["nvmlDeviceSetFanSpeed_v2"].side_effect = [
        None, hardware.pynvml.NVMLError(hardware.pynvml.NVML_ERROR_UNKNOWN),
    ]
    with pytest.raises(ValueError, match="Failed to set fan speed"):
        manager.set_fan_auto(0)
    assert not manager._auto_fans
    assert manager._fan_thread is None
    assert nvml["nvmlDeviceSetDefaultFanSpeed_v2"].call_args_list == [
        call("gpu0", 0), call("gpu0", 1),
    ]


def test_shutdown_stops_controller_before_resetting_all_fans(fans):
    manager, nvml, tick = fans
    manager.set_fan_auto(0)
    manager.shutdown()
    assert manager._fan_stop.is_set()
    manager._fan_thread.join.assert_called_once()
    assert not manager._auto_fans
    assert nvml["nvmlDeviceSetDefaultFanSpeed_v2"].call_args_list == [
        call("gpu0", 0), call("gpu0", 1),
    ]


def test_fanless_gpu_rejects_auto_without_writes(fans):
    manager, nvml, tick = fans
    nvml["nvmlDeviceGetNumFans"].return_value = 0
    with pytest.raises(ValueError, match="does not support"):
        manager.set_fan_auto(0)
    nvml["nvmlDeviceSetFanSpeed_v2"].assert_not_called()
