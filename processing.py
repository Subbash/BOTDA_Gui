import numpy as np

from adq_controller import ADQController


def acquire_single_trace_raw_ch0(controller: ADQController) -> np.ndarray:
    """Acquire a single trace from channel 0 only for probe."""
    controller.arm()
    ch0, _ = controller.acquire_once()
    controller.disarm()
    return ch0.astype(np.float32)


def average_spatial_samples(trace: np.ndarray, group_size: int) -> np.ndarray:
    n = len(trace)
    usable = n - (n % group_size)
    reshaped = trace[:usable].reshape(-1, group_size)
    return reshaped.mean(axis=1)


def process_trace(trace: np.ndarray, cfg: dict, valid_window_us: float, apply_dc: bool = True):
    fs = float(cfg["sample_rate"])
    prf = float(cfg["PRF"])
    pw = float(cfg.get("pulselength"))

    group_size = int(pw * fs * 1e-9)

    samples_per_period = int(round(fs / prf))
    valid_samples = int(round(valid_window_us * fs))

    n = trace.size
    diff = trace[:n].astype(np.float32)

    total_samples = diff.size
    n_periods = total_samples // samples_per_period
    if n_periods == 0:
        raise RuntimeError("Not enough samples for a single PRF period.")
    use_samples = n_periods * samples_per_period
    diff = diff[:use_samples]

    periods = diff.reshape(n_periods, samples_per_period)
    valid_samples = min(valid_samples, samples_per_period)
    periods_window = periods[:, :valid_samples]

    processed = periods_window.mean(axis=0, dtype=np.float64).astype(np.float32)
    processed = average_spatial_samples(processed, group_size)

    if apply_dc:
        tail_region = periods[:, valid_samples:]
        if tail_region.size == 0:
            raise RuntimeError("No tail samples available for DC estimation.")
        dc_offset = np.mean(tail_region)
        processed = processed - dc_offset

    meta_info = dict(
        sample_rate=fs,
        prf=prf,
        samples_per_period=samples_per_period,
        valid_samples=valid_samples,
        valid_window_us=valid_window_us,
        raw_length=n,
        usable_samples=use_samples,
        n_periods=n_periods,
        periods_window_shape=periods_window.shape,
        processed_shape=processed.shape,
    )

    return processed, meta_info


def bulk_process_traces(raw_traces, cfg, valid_window_us, apply_dc=True):
    """Bulk process a list of raw traces using process_trace()."""
    processed_traces = []
    meta_list = []
    for raw in raw_traces:
        proc, meta = process_trace(raw, cfg, valid_window_us, apply_dc=apply_dc)
        processed_traces.append(proc)
        meta_list.append(meta)
    return np.vstack(processed_traces), meta_list
