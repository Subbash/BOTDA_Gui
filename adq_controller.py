import ctypes as ct
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pyadq


class ADQController:
    """Manage ADQ device configuration and acquisition."""

    def __init__(self, config: Dict) -> None:
        self.config = self._calculate_derived_params(config)
        self.acu = None
        self.dev = None

    def connect(self) -> None:
        """Initialize ADQ control unit and configure device."""
        self.acu = pyadq.ADQControlUnit()
        self.acu.ADQControlUnit_EnableErrorTrace(pyadq.LOG_LEVEL_INFO, ".")

        dev_list = self.acu.ListDevices()
        if not dev_list:
            raise RuntimeError("No ADQ devices found")

        idx = next(
            (
                i
                for i, info in enumerate(dev_list)
                if info.ProductID in [pyadq.PID_ADQ14, pyadq.PID_ADQ7, pyadq.PID_ADQ8]
            ),
            -1,
        )
        if idx < 0:
            raise RuntimeError("No supported ADQ device found (expect ADQ14/7/8)")

        self.dev = self.acu.SetupDevice(idx)

        if not self.dev.ADQ_SetTriggerMode(self.config["trigger_mode"]):
            raise RuntimeError("Failed to set trigger mode")

        self.dev.ADQ_SetInternalTriggerPeriod(self.config["trigger_period"])
        self.dev.ADQ_SetupTriggerOutput(0, 5, self.config["pulselength"], 0)
        self.dev.ADQ_SetSampleSkip(self.config["sample_skip"])
        self.dev.ADQ_SetPreTrigSamples(self.config["pretrigger"])
        self.dev.ADQ_SetTriggerDelay(self.config["trigger_delay"])
        self.dev.ADQ_MultiRecordSetChannelMask(self.config["channel_mask"])
        self.dev.ADQ_MultiRecordSetup(1, self.config["samples_per_record"])

        print("ADQ connected and configured.")

    def set_trigger_output(
        self, level_v: int = 5, pulse_len_ns: Optional[int] = None, invert: int = 0
    ) -> None:
        """Enable/shape the front-panel trigger output (TTL)."""
        self._ensure_connected()
        if pulse_len_ns is None:
            pulse_len_ns = self.config["pulselength"]
        self.dev.ADQ_SetupTriggerOutput(0, level_v, pulse_len_ns, invert)

    def arm(self) -> None:
        """Arm the device for acquisition."""
        self._ensure_connected()
        self.dev.ADQ_DisarmTrigger()
        self.dev.ADQ_ArmTrigger()

    def disarm(self) -> None:
        """Disarm the device."""
        self._ensure_connected()
        self.dev.ADQ_DisarmTrigger()

    def acquire_once(self) -> Tuple[np.ndarray, np.ndarray]:
        """Acquire one record and return data arrays (mV)."""
        self._ensure_connected()
        self.dev.ADQ_DisarmTrigger()
        self.dev.ADQ_ArmTrigger()

        n_samples = self.config["samples_per_record"]
        n_ch = self.config["number_of_channels"]

        target_buffers = (ct.POINTER(ct.c_int16 * n_samples) * n_ch)()
        for i in range(n_ch):
            target_buffers[i] = ct.pointer((ct.c_int16 * n_samples)())

        header_list = (pyadq.structs._ADQRecordHeader * 1)()
        target_buffers_vp = ct.cast(target_buffers, ct.POINTER(ct.c_void_p))

        start = time.time()
        while True:
            if self.dev.ADQ_GetAcquiredRecords() > 0:
                ok = self.dev.ADQ_GetDataWHTS(
                    target_buffers_vp,
                    ct.cast(header_list, ct.c_void_p),
                    None,
                    n_samples,
                    2,
                    0,
                    1,
                    self.config["channel_mask"],
                    0,
                    n_samples,
                    0x00,
                )
                if not ok:
                    raise RuntimeError("Data acquisition failed")
                break

            if (time.time() - start) > self.config["timeout_seconds"]:
                raise TimeoutError("Data acquisition timeout")

        data0_np = np.frombuffer(target_buffers[0].contents, dtype=np.int16, count=n_samples)
        data1_np = np.frombuffer(target_buffers[1].contents, dtype=np.int16, count=n_samples)

        data0_mv = data0_np * 0.15 + 95
        data1_mv = data1_np * 0.15 + 95
        return data0_mv, data1_mv

    def close(self) -> None:
        """Tear down multi-record and disable trigger output."""
        if self.dev is not None:
            self.dev.ADQ_SetupTriggerOutput(0, 0, self.config["pulselength"], 0)
            self.dev.ADQ_MultiRecordClose()

    def _ensure_connected(self) -> None:
        if self.dev is None:
            raise RuntimeError("ADQ device not connected. Call connect() first.")

    @staticmethod
    def _calculate_derived_params(cfg: Dict) -> Dict:
        cfg = dict(cfg)
        cfg.setdefault("sample_rate", 100e6)
        cfg.setdefault("sample_length", 100e6)
        cfg.setdefault("PRF", 5000)
        cfg.setdefault("fiblen_actual", 10000)
        cfg.setdefault("number_of_channels", 2)
        cfg.setdefault("pretrigger", 0)
        cfg.setdefault("trigger_delay", 80)
        cfg.setdefault("pulselength", 200)
        cfg.setdefault("trigger_mode", pyadq.ADQ_INTERNAL_TRIGGER_MODE)
        cfg.setdefault("timeout_seconds", 10)
        cfg.setdefault("filter_size", 1)

        cfg["samples_per_record"] = int(cfg["sample_length"])
        cfg["sample_skip"] = int(1e9 / cfg["sample_rate"])
        cfg["trigger_period"] = int(1 / cfg["PRF"] / 1e-9)
        cfg["channel_mask"] = 2 ** cfg["number_of_channels"] - 1
        return cfg
