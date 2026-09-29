#!/usr/bin/env python3
"""
Detailed Pause Annotator
========================
Semi-automated annotation of the internal structure of speech pauses.

    Step 1  PauseDetect    Segments each sound file into IPUs and pauses.
    Step 2  BreathDetect   Finds breath noises inside the pauses and adds a copy
                           of the result (AlignTier) for manual correction.
    Step 3  ClickDetect    Finds clicks inside the pauses (point tier).

Each step can be switched on or off. If Step 1 is off, Steps 2 and 3 read the
pauses from an existing pause/IPU tier in the user's own TextGrids.

Start the program with:

    python pause_annotator.py

Requirements: praat-parselmouth, numpy, scipy, tgt (see requirements.txt).
All settings are described in README.md.
"""

import bisect
import math
import queue
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parselmouth
import tgt
from parselmouth.praat import call
from scipy.signal import butter, sosfilt

__version__ = "1.1.0"

# ═══════════════════════════════════════════════════════════════════════════════
# Fixed names
# ═══════════════════════════════════════════════════════════════════════════════

TIER_PAUSE = "PauseDetect"
TIER_BREATH = "BreathDetect"
TIER_ALIGN = "AlignTier"
TIER_CLICK = "ClickDetect"

LABEL_IPU = "ipu"
LABEL_PAUSE = "pause"

OUTPUT_SUBFOLDER = "annotated"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".aiff", ".aif", ".aifc", ".flac", ".ogg"}

# Long files are filtered in overlapping chunks to keep memory use low.
FILTER_CHUNK_S = 60.0
FILTER_PAD_S = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Settings (defaults and GUI layout are generated from these lists)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Param:
    key: str
    label: str
    default: object
    kind: str = "float"          # float | str | bool | choice
    advanced: bool = False
    group: str = ""              # heading of the advanced-settings box
    choices: tuple = ()


STEP1_PARAMS = [
    Param("threshold_db", "Silence threshold (dB below peak)", 30.0),
    Param("min_silence", "Minimum pause duration (s)", 0.2),
    Param("min_sounding", "Minimum IPU duration (s)", 0.1),
    Param("burst_correction", "Move short bursts before speech onset into the pause", True, "bool"),
    Param("ghost_filter", "Relabel IPUs without voicing as pauses", True, "bool"),
    # advanced
    Param("filter_low", "Band-pass low cut-off (Hz, 0 = none)", 80.0, advanced=True, group="Filter and intensity"),
    Param("filter_high", "Band-pass high cut-off (Hz, 0 = none)", 8000.0, advanced=True, group="Filter and intensity"),
    Param("filter_smooth", "Hann smoothing (Hz)", 80.0, advanced=True, group="Filter and intensity"),
    Param("intensity_min_pitch", "Intensity minimum pitch (Hz)", 100.0, advanced=True, group="Filter and intensity"),
    Param("time_step", "Intensity time step (s, 0 = auto)", 0.008, advanced=True, group="Filter and intensity"),
    Param("burst_max_dur", "Maximum burst duration (s)", 0.060, advanced=True, group="Burst correction"),
    Param("burst_onset_window", "Search window after IPU onset (s)", 0.300, advanced=True, group="Burst correction"),
    Param("burst_falloff_window", "Window for energy drop (s)", 0.030, advanced=True, group="Burst correction"),
    Param("burst_falloff_margin", "Energy drop: dB above threshold", 4.0, advanced=True, group="Burst correction"),
    Param("min_voiced", "Minimum voiced time per IPU (s)", 0.050, advanced=True, group="Voicing check"),
    Param("pitch_floor", "Pitch floor (Hz)", 50.0, advanced=True, group="Voicing check"),
    Param("pitch_ceiling", "Pitch ceiling (Hz)", 700.0, advanced=True, group="Voicing check"),
    Param("pitch_time_step", "Pitch time step (s)", 0.010, advanced=True, group="Voicing check"),
    Param("pitch_silence_threshold", "Silence threshold", 0.01, advanced=True, group="Voicing check"),
    Param("pitch_voicing_threshold", "Voicing threshold", 0.20, advanced=True, group="Voicing check"),
    Param("octave_cost", "Octave cost", 0.01, advanced=True, group="Voicing check"),
    Param("octave_jump_cost", "Octave-jump cost", 0.20, advanced=True, group="Voicing check"),
    Param("voiced_unvoiced_cost", "Voiced/unvoiced cost", 0.10, advanced=True, group="Voicing check"),
]

EXISTING_PARAMS = [
    Param("pause_tier", "Pause tier name", TIER_PAUSE, "str"),
    Param("pause_label", "Pause label", LABEL_PAUSE, "str"),
]

STEP2_PARAMS = [
    Param("threshold_mode", "Threshold mode", "absolute", "choice", choices=("absolute", "relative")),
    Param("abs_threshold", "Absolute threshold (dB)", 20.0),
    Param("rel_threshold", "Relative threshold (dB below file peak)", 50.0),
    Param("min_breath", "Minimum breath duration (s)", 0.2),
    Param("breath_label", "Breath label", "breath", "str"),
    Param("sil_label", "Silence label", "sil", "str"),
    # advanced
    Param("band_low", "Band-pass low cut-off (Hz)", 300.0, advanced=True, group="Filter and intensity"),
    Param("band_high", "Band-pass high cut-off (Hz)", 3000.0, advanced=True, group="Filter and intensity"),
    Param("band_smooth", "Hann smoothing (Hz)", 100.0, advanced=True, group="Filter and intensity"),
    Param("intensity_min_pitch", "Intensity minimum pitch (Hz)", 100.0, advanced=True, group="Filter and intensity"),
    Param("window_step", "Analysis step (s)", 0.010, advanced=True, group="Detection"),
    Param("bridge_gap", "Bridge gaps shorter than (s, 0 = off)", 0.0, advanced=True, group="Detection"),
]

STEP3_PARAMS = [
    Param("threshold_fraction", "Threshold (fraction of loudest frame)", 0.15),
    Param("min_dur_ms", "Minimum click duration (ms)", 0.5),
    Param("max_dur_ms", "Maximum click duration (ms)", 10.0),
    Param("click_label", "Click label", "click", "str"),
    # advanced
    Param("freq_low", "Band-pass low cut-off (Hz)", 3000.0, advanced=True, group="Filter"),
    Param("freq_high", "Band-pass high cut-off (Hz)", 10000.0, advanced=True, group="Filter"),
    Param("filter_order", "Butterworth filter order", 4.0, advanced=True, group="Filter"),
    Param("frame_ms", "Frame length (ms)", 1.0, advanced=True, group="Detection"),
    Param("hop_ms", "Frame hop (ms)", 0.5, advanced=True, group="Detection"),
    Param("merge_gap_ms", "Merge bursts closer than (ms)", 2.0, advanced=True, group="Detection"),
    Param("cluster_ms", "Cluster clicks within (ms)", 30.0, advanced=True, group="Detection"),
]


def defaults(params):
    return {p.key: p.default for p in params}


class StepError(Exception):
    """A problem that stops one step for one file (reported in the log)."""


# ═══════════════════════════════════════════════════════════════════════════════
# General helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_sound(path):
    sound = parselmouth.Sound(str(path))
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()
    return sound


def filter_in_chunks(sound, filter_fn):
    """Apply filter_fn (Sound -> Sound of equal length) to the whole sound.

    Short sounds are filtered in one go. Long sounds are cut into chunks with
    overlapping padding, so that FFT-based filters do not need the whole file
    in memory at once. The padding is discarded when the chunks are joined.
    """
    if sound.duration <= FILTER_CHUNK_S + 2 * FILTER_PAD_S:
        return filter_fn(sound)

    sr = sound.sampling_frequency
    values = sound.values[0]
    n = len(values)
    chunk_n = int(FILTER_CHUNK_S * sr)
    pad_n = int(FILTER_PAD_S * sr)
    out = np.empty(n)
    for a in range(0, n, chunk_n):
        b = min(n, a + chunk_n)
        pa, pb = max(0, a - pad_n), min(n, b + pad_n)
        filtered = filter_fn(parselmouth.Sound(values[pa:pb], sr)).values[0]
        out[a:b] = filtered[a - pa: a - pa + (b - a)]
    return parselmouth.Sound(out, sr, sound.xmin)


def merge_equal_labels(segments):
    """Merge neighbouring segments that carry the same label."""
    merged = []
    for seg in segments:
        if merged and seg["label"] == merged[-1]["label"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return merged


def segments_to_interval_tier(name, segments, start, end):
    """Convert contiguous segments to a tgt IntervalTier spanning start..end."""
    tier = tgt.core.IntervalTier(start, end, name)
    if not segments:
        tier.add_interval(tgt.core.Interval(start, end, ""))
        return tier
    bounds = [start] + [s["end"] for s in segments[:-1]] + [end]
    intervals = []
    for i, seg in enumerate(segments):
        s, e = bounds[i], bounds[i + 1]
        if e - s > 1e-9:
            intervals.append(tgt.core.Interval(s, e, seg["label"]))
    tier.add_intervals(intervals)
    return tier


def read_textgrid(path):
    """Read a TextGrid in any Praat format and text encoding."""
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xfe\xff", b"\xff\xfe"):
        encoding = "utf-16"
    elif raw[:3] == b"\xef\xbb\xbf":
        encoding = "utf-8-sig"
    else:
        try:
            raw.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            encoding = "latin-1"
    return tgt.io.read_textgrid(str(path), encoding=encoding, include_empty_intervals=True)


def pause_segments_from_tier(tier, pause_label, start, end):
    """Build contiguous segments from an existing tier.

    Returns segments with a boolean 'pause' key. Every interval whose label
    matches pause_label (case-insensitive, surrounding spaces ignored) is a
    pause; everything else, including gaps in the tier, counts as IPU.
    """
    target = pause_label.strip().lower()
    segments = []
    cursor = start
    for iv in sorted(tier.intervals, key=lambda i: i.start_time):
        if iv.start_time > cursor + 1e-9:
            segments.append({"start": cursor, "end": iv.start_time, "pause": False})
        is_pause = iv.text.strip().lower() == target
        segments.append({"start": iv.start_time, "end": iv.end_time, "pause": is_pause})
        cursor = iv.end_time
    if end > cursor + 1e-9:
        segments.append({"start": cursor, "end": end, "pause": False})
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 – PauseDetect
# ═══════════════════════════════════════════════════════════════════════════════

def find_onset_bursts(segments, times, values, threshold, p):
    """Short energy bursts near the start of an IPU that are followed by a
    drop in energy (typically clicks right before speech onset).

    Each burst is extended forward to the point where the energy rises above
    the threshold again, i.e. to the actual speech onset.
    """
    sounding = values >= threshold
    runs = []
    run_start, state = times[0], sounding[0]
    for i in range(1, len(times)):
        if sounding[i] != state:
            if state:
                runs.append((float(run_start), float(times[i])))
            run_start, state = times[i], sounding[i]
    if state:
        runs.append((float(run_start), float(times[-1])))

    starts = [s["start"] for s in segments]
    bursts = []
    for b_start, b_end in runs:
        if b_end - b_start >= p["burst_max_dur"]:
            continue
        mid = (b_start + b_end) / 2.0
        idx = max(0, bisect.bisect_left(starts, mid) - 1)
        seg = segments[idx]
        if not (seg["start"] <= mid <= seg["end"]) or seg["label"] != LABEL_IPU:
            continue
        if mid - seg["start"] > p["burst_onset_window"]:
            continue
        after = values[(times > b_end) & (times <= b_end + p["burst_falloff_window"])]
        after = after[np.isfinite(after)]
        if len(after) == 0 or np.min(after) > threshold + p["burst_falloff_margin"]:
            continue
        extended_end = b_end
        ahead = np.where((times > b_end) & (times <= seg["end"]))[0]
        for k in ahead:
            if values[k] >= threshold:
                extended_end = float(times[k])
                break
        bursts.append({"start": b_start, "end": extended_end})

    bursts.sort(key=lambda b: b["start"])
    merged = []
    for b in bursts:
        if merged and b["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], b["end"])
        else:
            merged.append(dict(b))
    return merged


def apply_burst_correction(segments, bursts, tolerance=0.001):
    """Move bursts found at the start of an IPU into the preceding pause."""
    segs = [dict(s) for s in segments]
    result = []
    for seg in segs:
        if seg["label"] == LABEL_IPU:
            for b in bursts:
                start_match = abs(seg["start"] - b["start"]) <= tolerance
                end_match = abs(seg["end"] - b["end"]) <= tolerance
                if start_match and end_match:
                    seg["label"] = LABEL_PAUSE
                    break
                if start_match:
                    if b["end"] < seg["end"]:
                        if result and result[-1]["label"] == LABEL_PAUSE:
                            result[-1]["end"] = b["end"]
                        else:
                            result.append({"start": seg["start"], "end": b["end"], "label": LABEL_PAUSE})
                        seg["start"] = b["end"]
                    break
        result.append(seg)
    return merge_equal_labels(result)


def apply_voicing_check(sound, segments, p):
    """Relabel IPUs with less than min_voiced seconds of voicing as pauses.

    The pitch tracker is deliberately liberal, so that quiet or creaky speech
    is still recognised as voiced and is not turned into a pause.
    """
    pitch = sound.to_pitch_ac(
        time_step=p["pitch_time_step"],
        pitch_floor=p["pitch_floor"],
        max_number_of_candidates=15,
        very_accurate=True,
        silence_threshold=p["pitch_silence_threshold"],
        voicing_threshold=p["pitch_voicing_threshold"],
        octave_cost=p["octave_cost"],
        octave_jump_cost=p["octave_jump_cost"],
        voiced_unvoiced_cost=p["voiced_unvoiced_cost"],
        pitch_ceiling=p["pitch_ceiling"],
    )
    times = pitch.xs()
    voiced = pitch.selected_array["frequency"] > 0
    voiced_cumsum = np.concatenate(([0], np.cumsum(voiced)))
    segs = [dict(s) for s in segments]
    for seg in segs:
        if seg["label"] != LABEL_IPU:
            continue
        i0 = np.searchsorted(times, seg["start"], side="left")
        i1 = np.searchsorted(times, seg["end"], side="right")
        voiced_time = (voiced_cumsum[i1] - voiced_cumsum[i0]) * p["pitch_time_step"]
        if voiced_time < p["min_voiced"]:
            seg["label"] = LABEL_PAUSE
    return merge_equal_labels(segs)


def run_pause_detection(sound, p):
    """Step 1. Returns segments labelled 'ipu' / 'pause' covering the sound.

    This reproduces Praat's "Sound: To TextGrid (silences)", with the
    band-pass filter exposed as a setting: the sound is filtered with a Hann
    band-pass (Praat uses 80-8000 Hz with 80 Hz smoothing), intensity is
    computed with the mean subtracted, and Praat's "Intensity: To TextGrid
    (silences)" finds the pauses. With the default filter, the result is
    identical to Praat's silences annotator. The burst correction works on
    the very same intensity contour, so its bursts line up exactly with the
    pause boundaries.
    """
    low, high = p["filter_low"], p["filter_high"]
    if low > 0 or high > 0:
        work = filter_in_chunks(sound, lambda s: call(s, "Filter (pass Hann band)", low, high, p["filter_smooth"]))
    else:
        work = sound
    try:
        intensity = work.to_intensity(minimum_pitch=p["intensity_min_pitch"],
                                      time_step=p["time_step"] or None, subtract_mean=True)
        grid = call(intensity, "To TextGrid (silences)", -abs(p["threshold_db"]),
                    p["min_silence"], p["min_sounding"], LABEL_PAUSE, LABEL_IPU)
    except parselmouth.PraatError as e:
        raise StepError(f"silence detection failed ({str(e).splitlines()[0]})")

    segments = []
    for i in range(1, call(grid, "Get number of intervals", 1) + 1):
        segments.append({"start": call(grid, "Get start time of interval", 1, i),
                         "end": call(grid, "Get end time of interval", 1, i),
                         "label": call(grid, "Get label of interval", 1, i)})

    if p["burst_correction"] and any(s["label"] == LABEL_PAUSE for s in segments):
        times = intensity.xs()
        values = intensity.values[0].copy()
        threshold = call(intensity, "Get maximum", 0, 0, "Parabolic") - abs(p["threshold_db"])
        if len(times):
            bursts = find_onset_bursts(segments, times, values, threshold, p)
            segments = apply_burst_correction(segments, bursts)

    if p["ghost_filter"]:
        segments = apply_voicing_check(sound, segments, p)
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 – BreathDetect
# ═══════════════════════════════════════════════════════════════════════════════

def find_breath_runs(intensity, start, end, threshold, p):
    """Above-threshold runs inside one pause, queried in fixed steps."""
    step = p["window_step"]
    dur = end - start
    n_windows = int(math.floor(dur / step + 1e-9))
    runs, run_start = [], None
    for k in range(n_windows):
        t = k * step
        db = intensity.get_value(start + t + step / 2.0)
        is_breath = db is not None and np.isfinite(db) and db >= threshold
        if is_breath and run_start is None:
            run_start = t
        elif not is_breath and run_start is not None:
            runs.append([run_start, t])
            run_start = None
    if run_start is not None:
        runs.append([run_start, dur])

    if p["bridge_gap"] > 0:
        bridged = []
        for r in runs:
            if bridged and r[0] - bridged[-1][1] < p["bridge_gap"]:
                bridged[-1][1] = r[1]
            else:
                bridged.append(r)
        runs = bridged

    return [(start + a, start + b) for a, b in runs if b - a >= p["min_breath"] - 1e-9]


def run_breath_detection(sound, segments, p):
    """Step 2. Returns BreathDetect segments (ipu / breath / sil)."""
    nyquist = sound.sampling_frequency / 2.0
    if p["band_low"] >= min(p["band_high"], nyquist):
        raise StepError(f"breath band {p['band_low']:.0f}–{p['band_high']:.0f} Hz "
                        f"does not fit the sampling rate")
    filtered = filter_in_chunks(
        sound, lambda s: call(s, "Filter (pass Hann band)", p["band_low"], p["band_high"], p["band_smooth"]))
    try:
        intensity = filtered.to_intensity(minimum_pitch=p["intensity_min_pitch"], subtract_mean=True)
    except parselmouth.PraatError as e:
        raise StepError(f"intensity analysis failed ({str(e).splitlines()[0]})")

    if p["threshold_mode"] == "relative":
        threshold = call(intensity, "Get maximum", 0, 0, "Parabolic") - abs(p["rel_threshold"])
    else:
        threshold = p["abs_threshold"]

    result = []
    for seg in segments:
        if not seg["pause"]:
            result.append({"start": seg["start"], "end": seg["end"], "label": LABEL_IPU})
            continue
        cursor = seg["start"]
        for b_start, b_end in find_breath_runs(intensity, seg["start"], seg["end"], threshold, p):
            if b_start > cursor + 1e-6:
                result.append({"start": cursor, "end": b_start, "label": p["sil_label"]})
            result.append({"start": b_start, "end": b_end, "label": p["breath_label"]})
            cursor = b_end
        if seg["end"] > cursor + 1e-6:
            result.append({"start": cursor, "end": seg["end"], "label": p["sil_label"]})
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 – ClickDetect
# ═══════════════════════════════════════════════════════════════════════════════

def frame_rms(x, frame_n, hop_n):
    """RMS of frames starting at 0, hop_n, 2*hop_n, ... (full frames only)."""
    if len(x) < frame_n:
        return np.empty(0)
    csum = np.concatenate(([0.0], np.cumsum(x * x)))
    starts = np.arange(0, len(x) - frame_n + 1, hop_n)
    power = (csum[starts + frame_n] - csum[starts]) / frame_n
    return np.sqrt(np.maximum(power, 0.0))


def loudest_frame(x, frame_n, hop_n, block_frames=200_000):
    """Maximum frame RMS over the whole signal, computed block by block."""
    peak = 0.0
    step = block_frames * hop_n
    for a in range(0, len(x), step):
        block = x[a: a + step + frame_n]
        e = frame_rms(block, frame_n, hop_n)
        if len(e):
            peak = max(peak, float(e[: block_frames].max()))
    return peak


def clicks_in_region(filtered, sr, region_start, region_end, abs_threshold, p):
    """Clicks inside one pause. Returns a list of (time, energy) at click peaks."""
    i_start, i_end = int(region_start * sr), int(region_end * sr)
    region = filtered[i_start:i_end]
    frame_n = max(1, int(p["frame_ms"] / 1000.0 * sr))
    hop_n = max(1, int(p["hop_ms"] / 1000.0 * sr))
    energies = frame_rms(region, frame_n, hop_n)
    if len(energies) == 0:
        return []
    positions = np.arange(len(energies)) * hop_n
    min_dur, max_dur = p["min_dur_ms"] / 1000.0, p["max_dur_ms"] / 1000.0

    def duration(cs, ce, runs_to_end=False):
        onset = positions[cs]
        offset = len(region) if runs_to_end else positions[min(ce, len(positions) - 1)] + frame_n
        return (offset - onset) / sr

    above = energies >= abs_threshold
    candidates, in_click, cs = [], False, 0
    for idx, flag in enumerate(above):
        if flag and not in_click:
            in_click, cs = True, idx
        elif not flag and in_click:
            in_click = False
            if min_dur <= duration(cs, idx) <= max_dur:
                candidates.append([cs, idx])
    if in_click and min_dur <= duration(cs, len(positions), runs_to_end=True) <= max_dur:
        candidates.append([cs, len(positions)])

    merge_gap_frames = max(1, int((p["merge_gap_ms"] / 1000.0) / (p["hop_ms"] / 1000.0)))
    merged = []
    for cs, ce in candidates:
        if merged and cs - merged[-1][1] < merge_gap_frames:
            merged[-1][1] = max(merged[-1][1], ce)
        else:
            merged.append([cs, ce])

    clicks = []
    for cs, ce in merged:
        if min_dur <= duration(cs, ce) <= max_dur:
            ce = min(ce, len(energies))
            peak_idx = cs + int(np.argmax(energies[cs:ce]))
            peak_time = region_start + (positions[peak_idx] + frame_n // 2) / sr
            clicks.append((peak_time, float(energies[peak_idx])))
    return clicks


def run_click_detection(sound, segments, p):
    """Step 3. Returns a sorted list of click times."""
    sr = sound.sampling_frequency
    nyquist = sr / 2.0
    low_n = max(p["freq_low"] / nyquist, 1e-6)
    high_n = min(p["freq_high"] / nyquist, 1.0 - 1e-6)
    if low_n >= high_n:
        raise StepError(f"click band {p['freq_low']:.0f}–{p['freq_high']:.0f} Hz "
                        f"does not fit the sampling rate ({sr:.0f} Hz)")
    sos = butter(int(round(p["filter_order"])), [low_n, high_n], btype="band", output="sos")
    filtered = sosfilt(sos, sound.values[0])

    frame_n = max(1, int(p["frame_ms"] / 1000.0 * sr))
    hop_n = max(1, int(p["hop_ms"] / 1000.0 * sr))
    reference = loudest_frame(filtered, frame_n, hop_n)
    if reference <= 0:
        return []
    abs_threshold = p["threshold_fraction"] * reference

    found = []
    for seg in segments:
        if seg["pause"]:
            found.extend(clicks_in_region(filtered, sr, seg["start"], seg["end"], abs_threshold, p))

    # Clicks close together form one cluster, marked at its loudest click.
    found.sort()
    gap = p["cluster_ms"] / 1000.0
    points, i = [], 0
    while i < len(found):
        cluster = [found[i]]
        while i + 1 < len(found) and found[i + 1][0] - cluster[0][0] <= gap:
            i += 1
            cluster.append(found[i])
        points.append(max(cluster, key=lambda c: c[1])[0])
        i += 1
    return points


# ═══════════════════════════════════════════════════════════════════════════════
# Per-file processing
# ═══════════════════════════════════════════════════════════════════════════════

def find_textgrid(folder, stem):
    for candidate in folder.iterdir():
        if candidate.suffix.lower() == ".textgrid" and candidate.stem.lower() == stem.lower():
            return candidate
    return None


def process_file(audio_path, settings, tick, stop_event):
    """Run the selected steps on one file and write the result.

    tick() is called after each completed step (for the progress bar).
    Returns a short summary string for the log.
    """
    run1, run2, run3 = settings["run1"], settings["run2"], settings["run3"]
    sound = load_sound(audio_path)
    summary = []

    # ── Pauses: detect (Step 1) or read from the user's TextGrid ──────────────
    if run1:
        tg = tgt.core.TextGrid()
        start, end = sound.xmin, sound.xmax
        pause_segs = run_pause_detection(sound, settings["step1"])
        tg.add_tier(segments_to_interval_tier(TIER_PAUSE, pause_segs, start, end))
        segments = [{"start": s["start"], "end": s["end"], "pause": s["label"] == LABEL_PAUSE}
                    for s in pause_segs]
        summary.append(f"{sum(s['pause'] for s in segments)} pauses")
        tick()
    else:
        tg_path = find_textgrid(settings["textgrid_folder"], audio_path.stem)
        if tg_path is None:
            raise StepError(f"no TextGrid named {audio_path.stem}.TextGrid "
                            f"in {settings['textgrid_folder']}")
        tg = read_textgrid(tg_path)
        tier_name = settings["existing"]["pause_tier"]
        if not tg.has_tier(tier_name):
            raise StepError(f"{tg_path.name} has no tier called '{tier_name}'")
        tier = tg.get_tier_by_name(tier_name)
        if not isinstance(tier, tgt.core.IntervalTier):
            raise StepError(f"tier '{tier_name}' in {tg_path.name} is not an interval tier")
        start, end = tg.start_time, tg.end_time
        segments = pause_segments_from_tier(tier, settings["existing"]["pause_label"], start, end)
        n_pauses = sum(s["pause"] for s in segments)
        if n_pauses == 0:
            raise StepError(f"tier '{tier_name}' has no intervals labelled "
                            f"'{settings['existing']['pause_label']}'")
        summary.append(f"{n_pauses} pauses read")

    def add_or_replace(tier):
        if tg.has_tier(tier.name):
            tg.delete_tier(tier.name)
        tg.add_tier(tier)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    if run2 and not stop_event.is_set():
        breath_segs = run_breath_detection(sound, segments, settings["step2"])
        add_or_replace(segments_to_interval_tier(TIER_BREATH, breath_segs, start, end))
        add_or_replace(segments_to_interval_tier(TIER_ALIGN, breath_segs, start, end))
        n_breaths = sum(1 for s in breath_segs if s["label"] == settings["step2"]["breath_label"])
        summary.append(f"{n_breaths} breaths")
        tick()

    # ── Step 3 ────────────────────────────────────────────────────────────────
    if run3 and not stop_event.is_set():
        click_times = run_click_detection(sound, segments, settings["step3"])
        tier = tgt.core.PointTier(start, end, TIER_CLICK)
        label = settings["step3"]["click_label"]
        tier.add_points([tgt.core.Point(t, label) for t in click_times if start <= t <= end])
        add_or_replace(tier)
        summary.append(f"{len(click_times)} clicks")
        tick()

    if stop_event.is_set():
        return None

    out_folder = settings["output_folder"]
    out_folder.mkdir(parents=True, exist_ok=True)
    tgt.io.write_to_file(tg, str(out_folder / f"{audio_path.stem}.TextGrid"),
                         format="long", encoding="utf-8")
    return ", ".join(summary)


def collect_audio_files(folder):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def run_batch(files, settings, messages, stop_event):
    """Worker thread: processes all files and reports through the queue."""
    steps_per_file = sum(settings[k] for k in ("run1", "run2", "run3"))
    total = max(1, len(files) * steps_per_file)
    done = 0
    n_ok, n_failed = 0, 0

    for n, audio_path in enumerate(files, 1):
        if stop_event.is_set():
            break
        messages.put(("status", f"File {n} of {len(files)}: {audio_path.name}"))
        file_start = done

        def tick():
            nonlocal done
            done += 1
            messages.put(("progress", done, total))

        try:
            result = process_file(audio_path, settings, tick, stop_event)
            if result is None:
                break
            n_ok += 1
            messages.put(("log", f"{audio_path.name}: {result}"))
        except StepError as e:
            n_failed += 1
            messages.put(("log", f"{audio_path.name}: skipped – {e}"))
        except Exception as e:  # keep going with the next file
            n_failed += 1
            first_line = str(e).splitlines()[0] if str(e) else type(e).__name__
            messages.put(("log", f"{audio_path.name}: skipped – {first_line}"))
        done = file_start + steps_per_file
        messages.put(("progress", done, total))

    messages.put(("finished", n_ok, n_failed, stop_event.is_set()))


# ═══════════════════════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════════════════════

PROGRAM_NAME = "Detailed Pause Annotator"

_INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|https?://[^\s)]+|\*[^*\s][^*]*\*)")


class HelpWindow:
    """Shows README.md inside the program, with simple Markdown formatting."""

    _current = None

    @classmethod
    def show(cls, root):
        if cls._current is not None and cls._current.window.winfo_exists():
            cls._current.window.deiconify()
            cls._current.window.lift()
            return
        cls._current = cls(root)

    def __init__(self, root):
        import tkinter as tk
        import tkinter.font as tkfont
        from tkinter import ttk

        self.window = tk.Toplevel(root)
        self.window.title(f"Help – {PROGRAM_NAME}")
        self.window.geometry("820x720")

        frame = ttk.Frame(self.window, padding=(10, 10, 10, 0))
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.text = tk.Text(frame, wrap="word", padx=16, pady=12, relief="flat",
                            background="white", foreground="black", cursor="arrow")
        self.text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, command=self.text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=scroll.set)
        ttk.Button(self.window, text="Close", command=self.window.destroy).pack(anchor="e", padx=10, pady=10)

        base = tkfont.nametofont("TkDefaultFont").actual()
        family, size = base["family"], abs(base["size"]) or 10
        mono = tkfont.nametofont("TkFixedFont").actual()["family"]
        t = self.text
        t.configure(font=(family, size + 1), spacing1=2, spacing3=2)
        t.tag_configure("h1", font=(family, size + 9, "bold"), spacing1=6, spacing3=10)
        t.tag_configure("h2", font=(family, size + 5, "bold"), spacing1=16, spacing3=6)
        t.tag_configure("h3", font=(family, size + 3, "bold"), spacing1=12, spacing3=4)
        t.tag_configure("bold", font=(family, size + 1, "bold"))
        t.tag_configure("italic", font=(family, size + 1, "italic"))
        t.tag_configure("code", font=(mono, size), background="#f0f0f0")
        t.tag_configure("codeblock", font=(mono, size), background="#f0f0f0",
                        lmargin1=16, lmargin2=16, spacing1=0, spacing3=0)
        t.tag_configure("item", lmargin1=12, lmargin2=30, spacing1=3)
        t.tag_configure("link", foreground="#1a5fb4", underline=True)
        t.tag_bind("link", "<Enter>", lambda e: t.configure(cursor="hand2"))
        t.tag_bind("link", "<Leave>", lambda e: t.configure(cursor="arrow"))
        self._link_count = 0

        path = Path(__file__).resolve().with_name("README.md")
        if path.exists():
            self.render(path.read_text(encoding="utf-8"))
        else:
            t.insert("end", "README.md was not found next to pause_annotator.py.\n\n"
                            "It is part of the download on GitHub; please place it in the same "
                            "folder as the program.")
        t.configure(state="disabled")

    # ── Markdown rendering (headings, lists, tables, code, bold, italics, links) ──
    def inline(self, line, extra=()):
        pos = 0
        for m in _INLINE.finditer(line):
            if m.start() > pos:
                self.text.insert("end", line[pos:m.start()], extra)
            token = m.group(0)
            if token.startswith("**"):
                self.text.insert("end", token[2:-2], ("bold",) + tuple(extra))
            elif token.startswith("`"):
                self.text.insert("end", token[1:-1], ("code",) + tuple(extra))
            elif token.startswith("["):
                label, url = token[1:].split("](", 1)
                self.insert_link(label, url[:-1], extra)
            elif token.startswith("http"):
                self.insert_link(token, token, extra)
            else:
                self.text.insert("end", token[1:-1], ("italic",) + tuple(extra))
            pos = m.end()
        self.text.insert("end", line[pos:], extra)

    def insert_link(self, label, url, extra):
        import webbrowser
        self._link_count += 1
        tag = f"link{self._link_count}"
        self.text.tag_bind(tag, "<Button-1>", lambda e, u=url: webbrowser.open(u))
        self.text.insert("end", label, ("link", tag) + tuple(extra))

    def table(self, rows):
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
        body = [r for r in cells[1:] if not all(set(c) <= set("-: ") for c in r)]
        for row in body:
            self.text.insert("end", "•  ", "item")
            self.inline(row[0], ("bold", "item") if not row[0].startswith("`") else ("item",))
            rest = [c for c in row[1:] if c and c != "–"]
            if rest:
                self.text.insert("end", ": ", "item")
                self.inline(" – ".join(rest), ("item",))
            self.text.insert("end", "\n", "item")
        self.text.insert("end", "\n")

    def render(self, markdown):
        lines = markdown.splitlines()
        previous = None          # kind of the last block, to add space after lists

        def gap_after_list():
            if previous == "list":
                self.text.insert("end", "\n")

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith("```"):
                gap_after_list()
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    self.text.insert("end", lines[i] + "\n", "codeblock")
                    i += 1
                self.text.insert("end", "\n")
                previous = "code"
            elif stripped.startswith("[!["):
                pass  # badge images cannot be shown here
            elif stripped.startswith("|"):
                gap_after_list()
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append(lines[i])
                    i += 1
                self.table(rows)
                previous = "table"
                continue
            elif stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                self.text.insert("end", stripped[level:].strip() + "\n", f"h{min(level, 3)}")
                previous = "heading"
            elif re.match(r"^[*-] ", stripped) or re.match(r"^\d+\. ", stripped):
                if stripped[0] in "*-":
                    marker, rest = "•", stripped[2:]
                else:
                    marker, rest = stripped.split(" ", 1)
                self.text.insert("end", marker + "  ", "item")
                self.inline(rest, ("item",))
                self.text.insert("end", "\n", "item")
                previous = "list"
            elif stripped:
                gap_after_list()
                self.inline(stripped)
                self.text.insert("end", "\n\n")
                previous = "paragraph"
            i += 1


class ParamPanel:
    """Widgets and variables for one list of parameters."""

    def __init__(self, parent, params, advanced_parent=None):
        import tkinter as tk
        from tkinter import ttk
        self.params = params
        self.vars = {}
        self.widgets = []

        basic = [p for p in params if not p.advanced]
        row = 0
        for p in basic:
            row = self._add_row(parent, p, row, tk, ttk)

        advanced = [p for p in params if p.advanced]
        if advanced and advanced_parent is not None:
            groups = []
            for p in advanced:
                if p.group not in groups:
                    groups.append(p.group)
            # Two columns of roughly equal height; each column keeps the
            # original group order, and the column with the first group is on the left.
            sizes = {g: sum(1 for p in advanced if p.group == g) for g in groups}
            columns = [[], []]
            for group in sorted(groups, key=lambda g: -sizes[g]):
                target = min(columns, key=lambda c: sum(sizes[g] for g in c))
                target.append(group)
            columns = [sorted(c, key=groups.index) for c in columns if c]
            columns.sort(key=lambda c: groups.index(c[0]))
            for col, col_groups in enumerate(columns):
                column = ttk.Frame(advanced_parent)
                column.grid(row=0, column=col, sticky="nw", padx=(0, 10))
                for group in col_groups:
                    box = ttk.LabelFrame(column, text=group, padding=(8, 4))
                    box.pack(anchor="nw", fill="x", pady=(0, 6))
                    r = 0
                    for p in advanced:
                        if p.group == group:
                            r = self._add_row(box, p, r, tk, ttk)

    def _add_row(self, parent, p, row, tk, ttk):
        if p.kind == "bool":
            var = tk.BooleanVar(value=p.default)
            w = ttk.Checkbutton(parent, text=p.label, variable=var)
            w.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
            self.widgets.append(w)
        else:
            var = tk.StringVar(value=str(p.default))
            label = ttk.Label(parent, text=p.label)
            label.grid(row=row, column=0, sticky="w", pady=2, padx=(0, 8))
            if p.kind == "choice":
                w = ttk.Combobox(parent, textvariable=var, values=p.choices, state="readonly", width=10)
            else:
                w = ttk.Entry(parent, textvariable=var, width=12)
            w.grid(row=row, column=1, sticky="w", pady=2)
            self.widgets += [label, w]
        self.vars[p.key] = var
        return row + 1

    def values(self, step_name, errors):
        out = {}
        for p in self.params:
            raw = self.vars[p.key].get()
            if p.kind == "float":
                try:
                    out[p.key] = float(str(raw).replace(",", "."))
                except ValueError:
                    errors.append(f"{step_name}: '{p.label}' must be a number.")
            elif p.kind == "str":
                out[p.key] = str(raw).strip()
            else:
                out[p.key] = raw
        return out

    def reset(self):
        for p in self.params:
            self.vars[p.key].set(p.default if p.kind == "bool" else str(p.default))

    def set_enabled(self, enabled):
        from tkinter import ttk
        for w in self.widgets:
            if isinstance(w, ttk.Combobox):
                w.configure(state="readonly" if enabled else "disabled")
            else:
                w.state(["!disabled"] if enabled else ["disabled"])


class ScrollableFrame:
    """A frame with a vertical scrollbar that appears only when needed.

    Widgets go into .inner; .outer is placed in the layout.
    """

    def __init__(self, parent, padding=10):
        import tkinter as tk
        from tkinter import ttk
        self.outer = ttk.Frame(parent)
        self.outer.rowconfigure(0, weight=1)
        self.outer.columnconfigure(0, weight=1)
        background = ttk.Style().lookup("TFrame", "background") or None
        self.canvas = tk.Canvas(self.outer, highlightthickness=0, borderwidth=0, height=200)
        if background:
            self.canvas.configure(background=background)
        self.vbar = ttk.Scrollbar(self.outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.inner = ttk.Frame(self.canvas, padding=padding)
        self.item = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", lambda e: self._update())
        self.canvas.bind("<Configure>", lambda e: self._update())
        self.canvas.bind("<Enter>", lambda e: self._bind_wheel(True))
        self.canvas.bind("<Leave>", lambda e: self._bind_wheel(False))

    def _update(self):
        req_w, req_h = self.inner.winfo_reqwidth(), self.inner.winfo_reqheight()
        self.canvas.configure(width=req_w, scrollregion=(0, 0, req_w, req_h))
        self.canvas.itemconfigure(self.item, width=max(req_w, self.canvas.winfo_width()))
        if req_h <= self.canvas.winfo_height():
            self.canvas.yview_moveto(0)
            self.vbar.grid_remove()
        else:
            self.vbar.grid()

    def needs_scrolling(self):
        return self.inner.winfo_reqheight() > self.canvas.winfo_height()

    def _bind_wheel(self, active):
        if active:
            self.canvas.bind_all("<MouseWheel>", self._on_wheel)
            self.canvas.bind_all("<Button-4>", self._on_wheel)
            self.canvas.bind_all("<Button-5>", self._on_wheel)
        else:
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.canvas.unbind_all(sequence)

    def _on_wheel(self, event):
        if not self.needs_scrolling():
            return
        if getattr(event, "num", None) == 4:
            units = -1
        elif getattr(event, "num", None) == 5:
            units = 1
        elif sys.platform == "darwin":
            units = -event.delta
        else:
            units = -int(event.delta / 120) or (-1 if event.delta > 0 else 1)
        self.canvas.yview_scroll(units, "units")

    def scroll_to(self, widget):
        """Scroll so that widget is at the top of the visible area, if possible."""
        self.canvas.update_idletasks()
        total = max(1, self.inner.winfo_reqheight())
        self.canvas.yview_moveto(widget.winfo_y() / total)


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root = root
        root.title(f"{PROGRAM_NAME} {__version__}")
        root.minsize(760, 560)

        self.messages = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None

        main = ttk.Frame(root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)

        # ── Input folder ──────────────────────────────────────────────────────
        ttk.Label(main, text="Sound files folder").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(main, text="Browse…", command=self.browse_input).grid(row=0, column=2)
        self.output_hint = ttk.Label(main, text="Results are saved to a subfolder called "
                                                f"'{OUTPUT_SUBFOLDER}' inside this folder.",
                                     foreground="gray40")
        self.output_hint.grid(row=1, column=1, sticky="w", padx=6, pady=(2, 8))

        # ── Step tabs ─────────────────────────────────────────────────────────
        notebook = ttk.Notebook(main)
        notebook.grid(row=2, column=0, columnspan=3, sticky="nsew")
        main.rowconfigure(2, weight=1)

        self.run_vars = {}
        self.panels = {}
        tab_specs = [
            ("run1", "step1", "Step 1: Pauses", "Run Step 1 (pause detection)", STEP1_PARAMS),
            ("run2", "step2", "Step 2: Breaths", "Run Step 2 (breath detection)", STEP2_PARAMS),
            ("run3", "step3", "Step 3: Clicks", "Run Step 3 (click detection)", STEP3_PARAMS),
        ]
        self.scrollers = []
        for run_key, step_key, title, check_text, params in tab_specs:
            page = ttk.Frame(notebook)
            notebook.add(page, text=title)
            scroller = ScrollableFrame(page)
            scroller.outer.pack(fill="both", expand=True)
            self.scrollers.append(scroller)
            tab = scroller.inner

            run_var = tk.BooleanVar(value=True)
            self.run_vars[run_key] = run_var
            ttk.Checkbutton(tab, text=check_text, variable=run_var,
                            command=self.update_states).grid(row=0, column=0, sticky="w", pady=(0, 8))

            basic = ttk.Frame(tab)
            basic.grid(row=1, column=0, sticky="nw")
            adv_toggle = ttk.Button(tab, text="Show advanced settings")
            adv_toggle.grid(row=3, column=0, sticky="w", pady=(12, 6))
            adv = ttk.Frame(tab)
            adv_toggle.configure(command=lambda f=adv, b=adv_toggle, sc=scroller:
                                 self.toggle_advanced(f, b, sc))
            self.panels[step_key] = ParamPanel(basic, params, adv)

            if step_key == "step1":
                self.build_existing_frame(tab)

        # The visible area fits the basic settings; advanced settings scroll.
        root.update_idletasks()
        self.basic_height = max(sc.inner.winfo_reqheight() for sc in self.scrollers)
        for sc in self.scrollers:
            sc.canvas.configure(height=self.basic_height)

        # ── Progress and log ──────────────────────────────────────────────────
        self.progress = ttk.Progressbar(main, mode="determinate", maximum=100)
        self.progress.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 2))
        self.status_var = tk.StringVar(value="Choose a folder with sound files, then click Run.")
        ttk.Label(main, textvariable=self.status_var).grid(row=4, column=0, columnspan=3, sticky="w")

        log_frame = ttk.Frame(main)
        log_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(6, 6))
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=5, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

        # ── Buttons ───────────────────────────────────────────────────────────
        buttons = ttk.Frame(main)
        buttons.grid(row=6, column=0, columnspan=3, sticky="ew")
        ttk.Button(buttons, text="Help", command=lambda: HelpWindow.show(root)).pack(side="left")
        ttk.Button(buttons, text="Restore defaults", command=self.restore_defaults).pack(side="left", padx=6)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="right")
        self.run_button = ttk.Button(buttons, text="Run", command=self.start)
        self.run_button.pack(side="right", padx=6)

        self.update_states()

    # ── Layout helpers ────────────────────────────────────────────────────────
    def build_existing_frame(self, tab):
        ttk, tk = self.ttk, self.tk
        box = ttk.LabelFrame(tab, text="Existing segmentation (used when Step 1 is off)", padding=8)
        box.grid(row=1, column=1, sticky="new", padx=(24, 0))
        tab.columnconfigure(1, weight=1)
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="Steps 2 and 3 read the pauses from this tier of your own TextGrids. "
                            "All other intervals count as IPUs.",
                  wraplength=360, foreground="gray40").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        fields = ttk.Frame(box)
        fields.grid(row=1, column=0, columnspan=3, sticky="w")
        self.panels["existing"] = ParamPanel(fields, EXISTING_PARAMS)

        self.tg_label = ttk.Label(box, text="TextGrid folder (empty = sound files folder)")
        self.tg_label.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 2))
        self.tg_var = tk.StringVar()
        self.tg_entry = ttk.Entry(box, textvariable=self.tg_var)
        self.tg_entry.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.tg_button = ttk.Button(box, text="Browse…", command=self.browse_textgrids)
        self.tg_button.grid(row=3, column=2, padx=(6, 0))

    def toggle_advanced(self, frame, button, scroller):
        if frame.grid_info():
            frame.grid_remove()
            button.configure(text="Show advanced settings")
            scroller.canvas.configure(height=self.basic_height)
        else:
            frame.grid(row=4, column=0, columnspan=2, sticky="nw")
            button.configure(text="Hide advanced settings")
            # Grow the settings area as far as the screen allows; scroll the rest.
            self.root.update_idletasks()
            other = self.root.winfo_height() - scroller.canvas.winfo_height()
            available = self.root.winfo_screenheight() - other - 120
            wanted = scroller.inner.winfo_reqheight()
            scroller.canvas.configure(height=max(self.basic_height, min(wanted, available)))
            scroller.scroll_to(button)

    def update_states(self):
        run1 = self.run_vars["run1"].get()
        self.panels["step1"].set_enabled(run1)
        self.panels["step2"].set_enabled(self.run_vars["run2"].get())
        self.panels["step3"].set_enabled(self.run_vars["run3"].get())
        self.panels["existing"].set_enabled(not run1)
        for w in (self.tg_label, self.tg_entry, self.tg_button):
            w.state(["disabled"] if run1 else ["!disabled"])

    def browse_input(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Folder with sound files")
        if folder:
            self.input_var.set(folder)

    def browse_textgrids(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Folder with TextGrids")
        if folder:
            self.tg_var.set(folder)

    def restore_defaults(self):
        for panel in self.panels.values():
            panel.reset()
        for var in self.run_vars.values():
            var.set(True)
        self.tg_var.set("")
        self.update_states()

    def write_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ── Running ───────────────────────────────────────────────────────────────
    def collect_settings(self):
        from tkinter import messagebox
        errors = []
        settings = {k: v.get() for k, v in self.run_vars.items()}
        if not settings["run1"] and not (settings["run2"] or settings["run3"]):
            errors.append("Switch on at least one step. Without Step 1, Step 2 or Step 3 is needed.")

        folder = Path(self.input_var.get().strip()) if self.input_var.get().strip() else None
        if folder is None or not folder.is_dir():
            errors.append("Choose an existing folder with sound files.")

        settings["step1"] = self.panels["step1"].values("Step 1", errors)
        settings["step2"] = self.panels["step2"].values("Step 2", errors)
        settings["step3"] = self.panels["step3"].values("Step 3", errors)
        settings["existing"] = self.panels["existing"].values("Existing segmentation", errors)

        if not errors:
            s1, s2, s3 = settings["step1"], settings["step2"], settings["step3"]
            if settings["run1"]:
                if s1["filter_low"] < 0 or s1["filter_high"] < 0 or s1["filter_smooth"] < 0:
                    errors.append("Step 1: filter frequencies cannot be negative.")
                elif s1["filter_high"] > 0 and s1["filter_low"] >= s1["filter_high"]:
                    errors.append("Step 1: the low cut-off must be below the high cut-off.")
            if settings["run2"]:
                if s2["band_low"] >= s2["band_high"]:
                    errors.append("Step 2: the low cut-off must be below the high cut-off.")
                if s2["window_step"] <= 0:
                    errors.append("Step 2: the analysis step must be greater than 0.")
                if not s2["breath_label"] or not s2["sil_label"]:
                    errors.append("Step 2: the breath and silence labels cannot be empty.")
            if settings["run3"]:
                if s3["freq_low"] >= s3["freq_high"]:
                    errors.append("Step 3: the low cut-off must be below the high cut-off.")
                if s3["min_dur_ms"] > s3["max_dur_ms"]:
                    errors.append("Step 3: the minimum click duration exceeds the maximum.")
                if s3["frame_ms"] <= 0 or s3["hop_ms"] <= 0:
                    errors.append("Step 3: frame length and hop must be greater than 0.")
            if not settings["run1"] and not settings["existing"]["pause_tier"]:
                errors.append("Enter the name of the pause tier in your TextGrids (Step 1 tab).")

        if errors:
            messagebox.showerror("Please check the settings", "\n".join(errors))
            return None

        settings["input_folder"] = folder
        settings["output_folder"] = folder / OUTPUT_SUBFOLDER
        tg_text = self.tg_var.get().strip()
        settings["textgrid_folder"] = Path(tg_text) if tg_text else folder
        if not settings["run1"] and not settings["textgrid_folder"].is_dir():
            messagebox.showerror("Please check the settings", "The TextGrid folder does not exist.")
            return None
        return settings

    def start(self):
        from tkinter import messagebox
        settings = self.collect_settings()
        if settings is None:
            return
        files = collect_audio_files(settings["input_folder"])
        if not files:
            messagebox.showinfo("No sound files",
                                "The folder contains no sound files "
                                f"({', '.join(sorted(AUDIO_EXTENSIONS))}).")
            return

        self.stop_event.clear()
        self.progress.configure(value=0)
        self.write_log(f"Processing {len(files)} file(s) in {settings['input_folder']}")
        self.run_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.worker = threading.Thread(target=run_batch,
                                       args=(files, settings, self.messages, self.stop_event),
                                       daemon=True)
        self.worker.start()
        self.root.after(100, self.poll)

    def stop(self):
        self.stop_event.set()
        self.status_var.set("Stopping after the current step…")
        self.stop_button.state(["disabled"])

    def poll(self):
        try:
            while True:
                msg = self.messages.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    self.progress.configure(value=100.0 * msg[1] / msg[2])
                elif kind == "status":
                    self.status_var.set(msg[1])
                elif kind == "log":
                    self.write_log(msg[1])
                elif kind == "finished":
                    _, n_ok, n_failed, stopped = msg
                    text = f"Done: {n_ok} file(s) annotated"
                    if n_failed:
                        text += f", {n_failed} skipped (see log)"
                    if stopped:
                        text = "Stopped. " + text
                    self.status_var.set(text + ".")
                    self.write_log(text + ".")
                    self.run_button.state(["!disabled"])
                    self.stop_button.state(["disabled"])
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.poll)


def main():
    try:
        import tkinter as tk
    except ImportError:
        sys.exit("Tkinter is not available in this Python installation. See README.md.")
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
