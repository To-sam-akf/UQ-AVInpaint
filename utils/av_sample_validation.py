import csv
from datetime import datetime
import os
from pathlib import Path

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


BAD_SAMPLE_LOG_NAME = "viai_av_bad_samples.csv"

BAD_SAMPLE_FIELDS = [
    "timestamp",
    "source",
    "phase",
    "split_name",
    "sample_path",
    "reason",
    "required_video_frames",
    "found_image_frames",
    "found_flow_x_frames",
    "found_flow_y_frames",
    "required_mel_frames",
    "found_mel_frames",
    "required_audio_steps",
    "found_audio_steps",
    "error_message",
]


class BadAVSampleError(ValueError):
    def __init__(self, reason, sample_path, record=None, error_message=None):
        self.reason = reason
        self.sample_path = str(sample_path)
        self.record = dict(record or {})
        self.record.setdefault("sample_path", self.sample_path)
        self.record.setdefault("reason", reason)
        self.record.setdefault("error_message", error_message or reason)
        super(BadAVSampleError, self).__init__(self.record["error_message"])


def _get(config, name, default=None):
    return getattr(config, name, default)


def default_bad_sample_log(data_root):
    return Path(data_root) / BAD_SAMPLE_LOG_NAME


def resolve_bad_sample_log(data_root, bad_sample_log=None):
    if bad_sample_log:
        return Path(bad_sample_log)
    return default_bad_sample_log(data_root)


def count_jpg_frames(directory):
    path = Path(directory)
    if not path.is_dir():
        return 0
    return sum(1 for item in path.glob("*.jpg") if item.is_file())


def av_window_requirements(config):
    sample_rate = int(_get(config, "sample_rate", 16000))
    hop_size = int(_get(config, "hop_size", 320))
    max_time_sec = _get(config, "max_time_sec", None)
    max_time_steps = _get(config, "max_time_steps", 64000)
    if max_time_sec is not None:
        max_time_steps = int(float(max_time_sec) * sample_rate)

    frame_stride = max(1, int(_get(config, "image_hope_size", _get(config, "frame_stride", 1))))
    visual_frame_count = int(_get(config, "visual_frame_count", 50))
    if visual_frame_count > 0:
        use_image_num = visual_frame_count
    else:
        if max_time_steps is None:
            raise ValueError("visual_frame_count must be positive when max_time_steps is None")
        max_time_second = float(max_time_steps) / float(sample_rate)
        use_image_num = int(np.floor(max_time_second / (0.04 * frame_stride)))
    if use_image_num <= 0:
        raise ValueError("use_image_num must be positive; check AV window settings")

    visual_frame_interval_sec = float(
        _get(config, "visual_frame_interval_sec", 0.04 * frame_stride)
    )
    mel_frames_per_visual_frame = visual_frame_interval_sec * sample_rate / hop_size
    mel_window_frames = int(round(use_image_num * mel_frames_per_visual_frame))
    required_audio_steps = mel_window_frames * hop_size
    last_offset = (use_image_num - 1) * frame_stride
    return {
        "frame_stride": frame_stride,
        "use_image_num": use_image_num,
        "last_offset": last_offset,
        "required_video_frames": last_offset + 1,
        "mel_frames_per_visual_frame": mel_frames_per_visual_frame,
        "required_mel_frames": mel_window_frames,
        "required_audio_steps": required_audio_steps,
    }


def _safe_npy_length(path):
    if not Path(path).exists():
        return None
    return int(np.load(path, mmap_mode="r").shape[0])


def inspect_av_sample(sample_dir, config):
    sample_dir = Path(sample_dir)
    requirements = av_window_requirements(config)
    record = {
        "sample_path": str(sample_dir),
        "required_video_frames": requirements["required_video_frames"],
        "found_image_frames": count_jpg_frames(sample_dir / "image_crop"),
        "found_flow_x_frames": count_jpg_frames(sample_dir / "flow_x_crop"),
        "found_flow_y_frames": count_jpg_frames(sample_dir / "flow_y_crop"),
        "required_mel_frames": requirements["required_mel_frames"],
        "found_mel_frames": _safe_npy_length(sample_dir / "mel.npy"),
        "required_audio_steps": requirements["required_audio_steps"],
        "found_audio_steps": _safe_npy_length(sample_dir / "raw_audio.npy"),
    }
    return record, requirements


def _raise_bad_sample(reason, sample_dir, record, message):
    raise BadAVSampleError(
        reason=reason,
        sample_path=sample_dir,
        record=record,
        error_message=message,
    )


def validate_av_sample(sample_dir, config):
    sample_dir = Path(sample_dir)
    record, requirements = inspect_av_sample(sample_dir, config)
    required_files = [sample_dir / "raw_audio.npy", sample_dir / "mel.npy"]
    required_dirs = [
        sample_dir / "image_crop",
        sample_dir / "flow_x_crop",
        sample_dir / "flow_y_crop",
    ]
    missing = [str(path) for path in required_files + required_dirs if not path.exists()]
    if missing:
        _raise_bad_sample(
            "missing_required_av_files",
            sample_dir,
            record,
            "Missing required AV sample files/directories: " + ", ".join(missing),
        )

    found_video_frames = min(
        record["found_image_frames"],
        record["found_flow_x_frames"],
        record["found_flow_y_frames"],
    )
    required_video_frames = int(record["required_video_frames"])
    if found_video_frames < required_video_frames:
        _raise_bad_sample(
            "insufficient_visual_frames",
            sample_dir,
            record,
            (
                f"Not enough aligned visual frames: need {required_video_frames}; "
                f"found image={record['found_image_frames']}, "
                f"flow_x={record['found_flow_x_frames']}, "
                f"flow_y={record['found_flow_y_frames']}"
            ),
        )

    found_mel_frames = record["found_mel_frames"]
    if found_mel_frames is None or found_mel_frames < record["required_mel_frames"]:
        _raise_bad_sample(
            "insufficient_mel_frames",
            sample_dir,
            record,
            (
                f"Not enough Mel frames: need {record['required_mel_frames']}; "
                f"found {found_mel_frames}"
            ),
        )

    found_audio_steps = record["found_audio_steps"]
    if found_audio_steps is None or found_audio_steps < record["required_audio_steps"]:
        _raise_bad_sample(
            "insufficient_audio_steps",
            sample_dir,
            record,
            (
                f"Not enough audio steps: need {record['required_audio_steps']}; "
                f"found {found_audio_steps}"
            ),
        )

    min_start = 25
    max_start = found_video_frames - 1 - requirements["last_offset"] - 25
    max_start_by_mel = int(
        np.floor(
            (found_mel_frames - requirements["required_mel_frames"])
            / requirements["mel_frames_per_visual_frame"]
        )
    )
    max_start = min(max_start, max_start_by_mel)
    if max_start < min_start:
        min_start = 0
        max_start = found_video_frames - 1 - requirements["last_offset"]
        max_start = min(max_start, max_start_by_mel)
    if max_start < min_start:
        _raise_bad_sample(
            "no_aligned_av_window",
            sample_dir,
            record,
            (
                f"No aligned AV window available: need video_frames={required_video_frames}, "
                f"mel_frames={record['required_mel_frames']}; "
                f"found video_frames={found_video_frames}, mel_frames={found_mel_frames}"
            ),
        )

    return record


def build_bad_sample_record(source, phase, split_name, sample_path, error):
    record = {field: "" for field in BAD_SAMPLE_FIELDS}
    record["timestamp"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    record["source"] = source or ""
    record["phase"] = phase or ""
    record["split_name"] = split_name or ""
    record["sample_path"] = str(sample_path or "")
    if isinstance(error, BadAVSampleError):
        for field in BAD_SAMPLE_FIELDS:
            if field in error.record and error.record[field] is not None:
                record[field] = error.record[field]
        record["sample_path"] = error.sample_path
        record["reason"] = error.reason
        record["error_message"] = str(error)
    else:
        record["reason"] = error.__class__.__name__
        record["error_message"] = str(error)

    for field, value in list(record.items()):
        if value is None:
            record[field] = ""
    return record


def append_bad_sample_log(log_path, record):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a+", encoding="utf-8", newline="") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            should_write_header = handle.tell() == 0
            writer = csv.DictWriter(handle, fieldnames=BAD_SAMPLE_FIELDS)
            if should_write_header:
                writer.writeheader()
            writer.writerow({field: record.get(field, "") for field in BAD_SAMPLE_FIELDS})
            handle.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return log_path


def log_bad_sample(data_root, bad_sample_log, source, phase, split_name, sample_path, error):
    log_path = resolve_bad_sample_log(data_root, bad_sample_log)
    record = build_bad_sample_record(source, phase, split_name, sample_path, error)
    return append_bad_sample_log(log_path, record)
