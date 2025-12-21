import time
import serial
from windfreak import SynthHD


class RFController:
    """Manage connection and control of a Windfreak SynthHD RF source."""

    def __init__(self, com_port: str, retries: int = 5, delay: float = 1.0) -> None:
        self.com_port = com_port
        self.retries = retries
        self.delay = delay
        self.device = None

    def connect(self) -> None:
        """Connect to the RF controller with retries."""
        for attempt in range(1, self.retries + 1):
            try:
                print(f"[Attempt {attempt}] Connecting to RF controller on {self.com_port}...")
                self.device = SynthHD(self.com_port)
                _ = self.device[0].frequency
                print(f"✅ Connected successfully on {self.com_port}")
                return
            except (serial.SerialException, UnicodeDecodeError, OSError) as exc:
                print(f"❌ Failed: {exc}")
                time.sleep(self.delay)

        raise RuntimeError(
            f"Could not connect to RF Synth on {self.com_port} after {self.retries} attempts"
        )

    def disconnect(self) -> None:
        """Safely close connection and release COM port."""
        if self.device is None:
            return

        try:
            if hasattr(self.device, "ser") and isinstance(self.device.ser, serial.Serial):
                self.device.ser.reset_input_buffer()
                self.device.ser.reset_output_buffer()
                self.device.ser.close()
                print("🔌 RF controller serial port closed.")
            self.device = None
            time.sleep(1.0)
            print("✅ RF controller disconnected and port released.")
        except Exception as exc:
            print(f"⚠️ Error while disconnecting: {exc}")

    def set_rf(self, channel: int, freq_ghz: float, power_dbm: float) -> None:
        """Set frequency and power on given channel."""
        self._ensure_connected()
        self.device[channel].frequency = freq_ghz * 1e9
        self.device[channel].power = power_dbm

    def enable_rf(self, channel: int = 0, enable: bool = True) -> None:
        """Enable or disable RF output on given channel."""
        self._ensure_connected()
        self.device[channel].enable = enable
        state = "ENABLED" if enable else "DISABLED"
        print(f"RF output {state} on channel {channel}")

    def get_reference(self) -> dict:
        """Get current reference source and frequency."""
        self._ensure_connected()
        try:
            mode = self.device.reference_mode
            freq = self.device.reference_frequency
            print("===== RF REFERENCE =====")
            print(f" Mode       : {mode}")
            print(f" Frequency  : {freq / 1e6:.3f} MHz")
            print("========================")
            return {"mode": mode, "frequency_hz": freq}
        except Exception as exc:
            print(f"⚠️ Error reading reference: {exc}")
            return {"mode": None, "frequency_hz": None}

    def set_reference(self, mode: str, freq_hz: float | None = None) -> None:
        """Set both reference mode and frequency."""
        self._ensure_connected()
        try:
            if mode not in self.device.reference_modes:
                raise ValueError(f"Invalid mode. Allowed: {self.device.reference_modes}")

            self.device.reference_mode = mode
            if freq_hz:
                self.device.reference_frequency = freq_hz
            elif mode == "internal 27mhz":
                self.device.reference_frequency = 27e6
            elif mode == "internal 10mhz":
                self.device.reference_frequency = 10e6
            elif mode == "external":
                self.device.reference_frequency = 10e6

            print(
                f"✅ Reference set to {mode} ({self.device.reference_frequency / 1e6:.3f} MHz)"
            )
        except Exception as exc:
            print(f"⚠️ Error setting reference: {exc}")

    def get_status(self, channel: int = 0) -> dict | None:
        """
        Return detailed RF status including reference info.

        Returns:
            dict: {
                "enabled": bool,
                "frequency_ghz": float,
                "power_dbm": float,
                "reference_mode": str,
                "reference_freq_mhz": float
            }
        """
        self._ensure_connected()
        try:
            freq_hz = self.device[channel].frequency
            power_dbm = self.device[channel].power
            enabled = self.device[channel].enable

            ref_info = self.get_reference()
            ref_mode = ref_info.get("mode")
            ref_freq = ref_info.get("frequency_hz")

            print("===== RF STATUS =====")
            print(f" RF Output : {'ON' if enabled else 'OFF'}")
            print(f" Frequency : {freq_hz / 1e9:.6f} GHz")
            print(f" Power     : {power_dbm:.2f} dBm")
            print(f" Ref Mode  : {ref_mode}")
            if ref_freq:
                print(f" Ref Freq  : {ref_freq / 1e6:.3f} MHz")
            print("======================")

            return {
                "enabled": enabled,
                "frequency_ghz": freq_hz / 1e9,
                "power_dbm": power_dbm,
                "reference_mode": ref_mode,
                "reference_freq_mhz": ref_freq / 1e6 if ref_freq else None,
            }
        except Exception as exc:
            print(f"⚠️ Could not read status: {exc}")
            return None

    def _ensure_connected(self) -> None:
        if self.device is None:
            raise RuntimeError("RF controller not connected. Call connect() first.")


if __name__ == "__main__":
    port = input("Enter COM port (e.g., COM5): ").strip()
    controller = RFController(port)
    controller.connect()
    controller.get_reference()
    controller.get_status()
    controller.set_reference("internal 27mhz")
    controller.get_status()
    controller.set_rf(0, 10.8, 0)
    controller.enable_rf(0, True)
    controller.get_status()
    controller.enable_rf(0, False)
    controller.disconnect()
