# sweep_rf_capture_once.py
import time
from dataclasses import dataclass

import numpy as np

from adq_controller import ADQController
from logger_utils import log_sweep_session
from plotter import plot_traces_3d_dist, save_traces
from processing import acquire_single_trace_raw_ch0, bulk_process_traces
from rf_controller import RFController

# ========== COMMENTS - EDIT THIS SECTION FOR EACH RUN ==========
PERSONAL_COMMENTS = """
Objective : Pulse width study

Group size formula added in processing

Laser: 1550.25nm
Pump : EOM - 130mv@30db attn - 17dbm
Probe : SSB - FBG BPF
Detector : 10db attenuated
PW  : 10ns, 1000
Test : 200Mhz sampling

"""
# ========== END COMMENTS ==========


@dataclass(frozen=True)
class RFParams:
    start_freq: float
    stop_freq: float
    step_mhz: float
    sweep_power_dbm: float
    com_port: str


@dataclass(frozen=True)
class SweepSettings:
    fiber_len_m: float
    dwell_time: float
    arm_before_each: bool
    inter_capture_delay: float
    settle_extra: float
    sample_rate_hz: float = 200e6
    averages: int = 1000
    pulselength_ns: int = 100
    save_dir: str = r"D:\Subbash\botda_exp\saved_plots"


DEFAULT_RF_PARAMS = RFParams(
    start_freq=10.700,
    stop_freq=11.0,
    step_mhz=1,
    sweep_power_dbm=-8.0,
    com_port="COM19",
)

DEFAULT_SWEEP_SETTINGS = SweepSettings(
    fiber_len_m=2e3,
    dwell_time=10.0,
    arm_before_each=True,
    inter_capture_delay=0.0,
    settle_extra=0.0,
)


class BOTDASweep:
    def __init__(self, rf_params: RFParams, sweep_settings: SweepSettings) -> None:
        self.rf_params = rf_params
        self.sweep_settings = sweep_settings
        self.adq_config = self.compute_adq_config(
            sweep_settings.fiber_len_m,
            sweep_settings.averages,
            sweep_settings.sample_rate_hz,
            sweep_settings.pulselength_ns
        )
        self.rf_controller = RFController(rf_params.com_port)
        self.adq_controller = ADQController(self.adq_config)

    @staticmethod
    def compute_adq_config(
        fiber_len_actual_m: float, 
        averages: int = 1000, 
        sample_rate: float = 200e6,
        pulselength: int = 100
    ) -> dict:
        valid_window_us = fiber_len_actual_m / 1e8
        tail_us = 0.2 * valid_window_us
        prp_us = valid_window_us + tail_us
        prp_s = prp_us
        total_periods = averages
        prf = 1 / prp_s
        sample_period_s = 1 / sample_rate
        samples_per_record = int(prp_s * total_periods / sample_period_s)

        return dict(
            sample_rate=sample_rate,
            sample_length=samples_per_record,
            PRF=prf,
            fiblen_actual=fiber_len_actual_m,
            number_of_channels=2,
            pretrigger=0,
            trigger_delay=0,
            pulselength=pulselength,
            timeout_seconds=1,
            valid_window_us=valid_window_us,
            tail_window_us=tail_us,
            prp_us=prp_us,
        )

    def sweep_rf_and_capture_once_raw(self, status_callback=None, progress_callback=None):
        start_freq = self.rf_params.start_freq
        stop_freq = self.rf_params.stop_freq
        step_mhz = self.rf_params.step_mhz
        sweep_power_dbm = self.rf_params.sweep_power_dbm

        self.rf_controller.connect()

        rf_status = self.rf_controller.get_status(channel=0)
        if rf_status and rf_status["enabled"]:
            current_freq = rf_status["frequency_ghz"]
            msg = f"RF already enabled at {current_freq:.6f} GHz"
            print(msg)
            if status_callback:
                status_callback(msg)

            if abs(current_freq - start_freq) < 1e-6:
                msg = "✅ Current frequency matches start frequency. Continuing..."
                print(msg)
                if status_callback:
                    status_callback(msg)
            else:
                msg = (
                    f"⚠️ Frequency differs ({current_freq:.6f} → {start_freq:.6f} GHz). "
                    "Updating and waiting 10 s..."
                )
                print(msg)
                if status_callback:
                    status_callback(msg)
                time.sleep(max(self.sweep_settings.dwell_time, 0))
                self.rf_controller.set_rf(channel=0, freq_ghz=start_freq, power_dbm=sweep_power_dbm)
        else:
            msg = "RF not enabled. Initializing..."
            print(msg)
            if status_callback:
                status_callback(msg)
            self.rf_controller.set_rf(channel=0, freq_ghz=start_freq, power_dbm=sweep_power_dbm)
            self.rf_controller.enable_rf(channel=0, enable=True)
            msg = "RF initialized and enabled."
            print(msg)
            if status_callback:
                status_callback(msg)
            time.sleep(max(self.sweep_settings.dwell_time, 0))
            if self.sweep_settings.settle_extra > 0:
                time.sleep(self.sweep_settings.settle_extra)

        self.adq_controller.connect()
        self.adq_controller.set_trigger_output(level_v=5)

        step_ghz = float(step_mhz) * 1e-3
        n_steps = int(((stop_freq - start_freq) / step_ghz) + 1)

        raw_traces = []
        freqs = []

        try:
            for i in range(n_steps):
                freq = start_freq + i * step_ghz
                self.rf_controller.set_rf(channel=0, freq_ghz=float(freq), power_dbm=sweep_power_dbm)

                raw_trace = acquire_single_trace_raw_ch0(self.adq_controller)

                msg = (
                    f"[{i + 1}/{n_steps}] RF -> {freq:.6f} GHz @ {sweep_power_dbm:.1f} dBm"
                    f" | Raw trace length {raw_trace.shape}"
                )
                print(f"\r{msg}", end="", flush=True)
                if status_callback:
                    status_callback(msg)
                if progress_callback:
                    progress_callback(i + 1, n_steps)

                raw_traces.append(raw_trace.astype(np.float32, copy=False))
                freqs.append(freq)

            return np.asarray(raw_traces, dtype=np.float32), np.asarray(freqs, dtype=float)
        finally:
            try:
                self.rf_controller.set_rf(
                    channel=0, freq_ghz=start_freq, power_dbm=sweep_power_dbm
                )
            except Exception:
                pass
            try:
                self.rf_controller.enable_rf(channel=0, enable=True)
            except Exception:
                pass

    def run(self, status_callback=None, progress_callback=None, plot=True, personal_comments=""):
        for k, v in self.adq_config.items():
            msg = f"{k}: {v}"
            print(msg)
            if status_callback:
                status_callback(msg)

        start_time = time.time()

        raw_traces, freqs = self.sweep_rf_and_capture_once_raw(status_callback, progress_callback)

        acq_end_time = time.time()
        elapsed_acq = acq_end_time - start_time
        msg = f"\nRaw acquisition completed in {elapsed_acq:.3f} seconds"
        print(msg)
        if status_callback:
            status_callback(msg)

        proc_start_time = time.time()
        processed_traces, meta_list = bulk_process_traces(
            raw_traces, self.adq_config, self.adq_config["valid_window_us"], apply_dc=True
        )

        proc_end_time = time.time()
        elapsed_proc = proc_end_time - proc_start_time
        msg = f"Processing completed in {elapsed_proc:.3f} seconds"
        print(msg)
        if status_callback:
            status_callback(msg)

        total_elapsed = proc_end_time - start_time
        msg = f"Total elapsed time (Acquisition + Processing): {total_elapsed:.3f} seconds"
        print(msg)
        if status_callback:
            status_callback(msg)

        n_steps, n_samples = processed_traces.shape
        fs = self.adq_config["sample_rate"]
        t_us = np.arange(n_samples) / fs * 1e6

        msg = f"Processing complete. Final traces shape = {processed_traces.shape}"
        print(msg)
        if status_callback:
            status_callback(msg)

        if plot:
            plot_traces_3d_dist(
                processed_traces,
                freqs,
                self.adq_config["fiblen_actual"],
                self.adq_config["pulselength"],
            )

        log_file = log_sweep_session(
            rf_params=self.rf_params.__dict__,
            adq_config=self.adq_config,
            traces=processed_traces,
            freqs=freqs,
            elapsed_time=total_elapsed,
            meta_info=meta_list[-1],
            save_dir=self.sweep_settings.save_dir,
            personal_comments=personal_comments,  # Use the passed comments
        )

        save_traces(
            processed_traces,
            freqs,
            t_us,
            save_dir=self.sweep_settings.save_dir,
            log_file=log_file,
        )

        return {
            "processed_traces": processed_traces,
            "freqs": freqs,
            "t_us": t_us,
            "log_file": log_file,
            "meta_info": meta_list[-1]
        }


if __name__ == "__main__":
    sweep = BOTDASweep(DEFAULT_RF_PARAMS, DEFAULT_SWEEP_SETTINGS)
    result = sweep.run()
    log_file = result["log_file"]

    try:
        print("\n--- Add comments for this sweep session ---")
        print("(Press ENTER twice to finish writing.)")
        user_comments = []
        while True:
            line = input()
            if line.strip() == "":
                break
            user_comments.append(line)
        comment_text = "\n".join(user_comments).strip()

        if comment_text:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n\n# ===== USER COMMENTS =====\n")
                f.write(comment_text + "\n")
            print(f"✅ Comments added to log file:\n{log_file}")
        else:
            print("No additional comments entered.")
    except Exception as exc:
        print(f"⚠️ Could not append comments: {exc}")
