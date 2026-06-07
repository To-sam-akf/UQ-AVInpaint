import argparse
import csv
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from utils.av_sample_validation import BadAVSampleError, validate_av_sample


YT_DLP_FORMAT = "mp4/bestvideo+bestaudio/best"
DOWNLOAD_PROGRESS_TEMPLATE = (
    "[download] %(progress._percent_str)s of %(progress._total_bytes_str)s "
    "at %(progress._speed_str)s ETA %(progress._eta_str)s"
)
BROWSER_COOKIE_SOURCES = {
    "firefox": {
        "binaries": ["firefox"],
        "dirs": [
            "~/.config/mozilla/firefox",
            "~/.mozilla/firefox",
            "~/.var/app/org.mozilla.firefox/config/mozilla/firefox",
            "~/.var/app/org.mozilla.firefox/.mozilla/firefox",
            "~/snap/firefox/common/.mozilla/firefox",
        ],
    },
    "librewolf": {
        "binaries": ["librewolf"],
        "dirs": [
            "~/.librewolf",
            "~/.var/app/io.gitlab.librewolf-community/config/librewolf",
        ],
    },
    "chrome": {
        "binaries": ["google-chrome", "chrome"],
        "dirs": [
            "~/.config/google-chrome",
        ],
    },
    "chromium": {
        "binaries": ["chromium", "chromium-browser"],
        "dirs": [
            "~/.config/chromium",
        ],
    },
    "edge": {
        "binaries": ["microsoft-edge", "microsoft-edge-stable", "edge"],
        "dirs": [
            "~/.config/microsoft-edge",
        ],
    },
    "brave": {
        "binaries": ["brave-browser", "brave"],
        "dirs": [
            "~/.config/BraveSoftware/Brave-Browser",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MUSICES data from MUSICES.json into VIAI training format."
    )
    parser.add_argument(
        "action",
        choices=["manifest", "stats", "download", "process", "splits", "all"],
        help="Preparation stage to run.",
    )
    parser.add_argument("--json", dest="json_path", default="/root/shared-nvme/data/MUSICES.json")
    parser.add_argument("--data-root", default="/root/shared-nvme/data")
    parser.add_argument("--video-dir", default="raw_videos")
    parser.add_argument(
        "--video-root",
        default=None,
        help="Optional filesystem root for raw videos. When set, downloaded videos are "
        "stored under this directory instead of `data-root/video-dir`.",
    )
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--manifest-name", default="musices_manifest.csv")
    parser.add_argument("--train-split-name", default="train_new_split.txt")
    parser.add_argument("--test-split-name", default="test_new_split.txt")
    parser.add_argument("--download-archive-name", default="musices_downloaded.txt")
    parser.add_argument("--stats-json-name", default="musices_download_stats.json")
    parser.add_argument("--stats-csv-name", default="musices_download_stats.csv")
    parser.add_argument("--test-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--fft-size", type=int, default=1280)
    parser.add_argument("--hop-size", type=int, default=320)
    parser.add_argument("--num-mels", type=int, default=80)
    parser.add_argument("--fmin", type=float, default=125.0)
    parser.add_argument("--fmax", type=float, default=7600.0)
    parser.add_argument("--min-level-db", type=float, default=-100.0)
    parser.add_argument("--ref-level-db", type=float, default=20.0)
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--clip-duration-sec",
        type=float,
        default=4.0,
        help="Duration of each AV training clip. The VIAI paper uses 4 seconds.",
    )
    parser.add_argument(
        "--clip-hop-sec",
        type=float,
        default=4.0,
        help="Hop size used to tile valid shots into AV clips. Default is non-overlapping 4-second windows.",
    )
    parser.add_argument(
        "--clip-mel-frames",
        type=int,
        default=200,
        help="Target Mel frame count per AV clip. The VIAI paper maps 4 seconds to an 80x200 spectrogram.",
    )
    parser.add_argument(
        "--visual-frame-count",
        type=int,
        default=50,
        help="Number of visual frames saved per AV clip. The VIAI paper maps 4 seconds to 50 video frames.",
    )
    parser.add_argument(
        "--max-clips-per-segment",
        type=int,
        default=None,
        help="Optional cap on the number of 4-second clips generated from each detected shot.",
    )
    parser.add_argument(
        "--max-clips-per-video",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of 4-second clips processed per source video. "
            "When set, clips are sampled deterministically from all valid shot windows before TV-L1 is computed."
        ),
    )
    parser.add_argument(
        "--trim-start-sec",
        type=float,
        default=6.0,
        help="Trim this many seconds from the start of each video. The VIAI paper removes the first 6 seconds.",
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=4.0,
        help="Minimum usable segment duration. VIAI trains on 4-second windows.",
    )
    parser.add_argument(
        "--shot-detection",
        dest="shot_detection",
        action="store_true",
        default=True,
        help="Split each video into shot-like segments before feature extraction.",
    )
    parser.add_argument(
        "--no-shot-detection",
        dest="shot_detection",
        action="store_false",
        help="Process each video as one sample. Use with --trim-start-sec 0 for legacy behavior.",
    )
    parser.add_argument(
        "--shot-diff-threshold",
        type=float,
        default=35.0,
        help="Mean grayscale frame-difference threshold used by the OpenCV shot detector.",
    )
    parser.add_argument(
        "--black-frame-threshold",
        type=float,
        default=10.0,
        help="Mean grayscale value below which a frame is counted as black.",
    )
    parser.add_argument(
        "--max-black-ratio",
        type=float,
        default=0.5,
        help="Skip segments whose sampled frames are black at or above this ratio.",
    )
    parser.add_argument(
        "--min-audio-rms",
        type=float,
        default=0.005,
        help="Skip segments whose audio RMS is below this threshold.",
    )
    parser.add_argument(
        "--flow-method",
        choices=["tvl1", "farneback"],
        default="tvl1",
        help="Optical-flow algorithm. VIAI paper uses TV-L1; Farneback is only for fallback smoke tests.",
    )
    parser.add_argument("--flow-clip", type=float, default=20.0)
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Show per-video progress bars for shot detection, frame reading, TV-L1 flow, and cropping.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable per-video progress bars.",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--abort-on-download-error", action="store_true")
    parser.add_argument("--yt-dlp-bin", default="yt-dlp")
    parser.add_argument(
        "--yt-dlp-extra-arg",
        action="append",
        default=[],
        help="Repeat to forward extra arguments to yt-dlp, for example "
        "`--yt-dlp-extra-arg=--cookies-from-browser --yt-dlp-extra-arg=firefox`.",
    )
    parser.add_argument(
        "--yt-dlp-js-runtime",
        default="auto",
        help="JavaScript runtime for yt-dlp YouTube challenges. Use `auto`, `none`, "
        "or a yt-dlp value such as `node`, `deno`, `bun`, or `quickjs:/path/to/qjs`.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    return parser.parse_args()


def require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for frame and optical-flow extraction. "
            "Install opencv-contrib-python-headless so mp4 decoding and TV-L1 "
            "optical flow are both available."
        ) from exc
    return cv2


def cv2_has_video_backend(cv2):
    try:
        build_info = cv2.getBuildInformation()
    except Exception:
        return True
    for line in build_info.splitlines():
        stripped = line.strip()
        if stripped.startswith("FFMPEG:") or stripped.startswith("GStreamer:"):
            value = stripped.split(":", 1)[1].strip().upper()
            if value.startswith("YES"):
                return True
    return False


def ensure_cv2_video_backend(cv2):
    if cv2_has_video_backend(cv2):
        return
    raise RuntimeError(
        "OpenCV was imported, but this build has no FFMPEG/GStreamer video backend. "
        "cv2.VideoCapture cannot decode mp4 files in this environment. Reinstall "
        "OpenCV in the same Python environment, for example: "
        "`python -m pip uninstall -y opencv-python opencv-python-headless "
        "opencv-contrib-python opencv-contrib-python-headless`, remove any leftover "
        "`cv2/` package directory if pip reports it is not empty, then install "
        "`python -m pip install --no-cache-dir numpy==1.22.4 "
        "opencv-contrib-python-headless==4.10.0.84`."
    )


class NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def update(self, amount=1):
        return None

    def close(self):
        return None


def make_progress(args, desc, total=None, unit="it"):
    if not getattr(args, "progress", True):
        return NullProgress()
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return NullProgress()
    if total is not None and total <= 0:
        return NullProgress()
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=False,
    )


def require_librosa():
    try:
        import librosa  # type: ignore
        import librosa.filters  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "librosa is required for audio and mel-spectrogram extraction. "
            "Install it with `uv add librosa && uv sync`."
        ) from exc
    return librosa


def require_numpy():
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "numpy is required for dataset preparation. "
            "Install it with `uv add numpy && uv sync`."
        ) from exc
    return np


def require_imageio_ffmpeg():
    try:
        import imageio_ffmpeg  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is required when system ffmpeg is unavailable. "
            "Install it with `uv add imageio-ffmpeg && uv sync`."
        ) from exc
    return imageio_ffmpeg


def load_records(json_path, max_videos=None):
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    videos = payload["videos"]
    records = []
    for instrument in sorted(videos.keys()):
        for youtube_id in videos[instrument]:
            records.append(
                {
                    "instrument": instrument,
                    "youtube_id": youtube_id,
                    "sample_key": f"{instrument}/{youtube_id}",
                }
            )
    if max_videos is not None:
        records = records[:max_videos]
    return records


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def manifest_path(data_root, manifest_name):
    return Path(data_root) / manifest_name


def stats_json_path(data_root, stats_json_name):
    return Path(data_root) / stats_json_name


def stats_csv_path(data_root, stats_csv_name):
    return Path(data_root) / stats_csv_name


def download_archive_path(data_root, download_archive_name):
    return Path(data_root) / download_archive_name


def download_failure_path(data_root):
    return Path(data_root) / "musices_download_failures.csv"


def process_failure_path(data_root):
    return Path(data_root) / "musices_process_failures.csv"


def resolved_video_root(data_root, video_dir, video_root=None):
    if video_root is not None:
        return Path(video_root)
    return Path(data_root) / video_dir


def video_output_path(data_root, video_dir, record, video_root=None):
    return resolved_video_root(data_root, video_dir, video_root) / record["instrument"] / f'{record["youtube_id"]}.mp4'


def sample_output_dir(data_root, processed_dir, record):
    return Path(data_root) / processed_dir / record["instrument"] / record["youtube_id"]


def segment_output_dir(data_root, processed_dir, record, segment_id):
    if segment_id is None:
        return sample_output_dir(data_root, processed_dir, record)
    return sample_output_dir(data_root, processed_dir, record) / f"shot_{segment_id:06d}"


def clip_output_dir(data_root, processed_dir, record, segment_id, clip_id):
    if segment_id is None:
        return sample_output_dir(data_root, processed_dir, record) / f"clip_{clip_id:06d}"
    return segment_output_dir(data_root, processed_dir, record, segment_id) / f"clip_{clip_id:06d}"


def video_url(record):
    return f'https://www.youtube.com/watch?v={record["youtube_id"]}'


def format_bytes(value):
    if value is None:
        return "unknown"
    if value < 1024:
        return f"{value} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        size /= 1024.0
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
    return f"{size:.2f} TiB"


def format_manifest_path(path, base_root):
    path = Path(path)
    base_root = Path(base_root)
    try:
        return path.relative_to(base_root).as_posix()
    except ValueError:
        return str(path)


def write_manifest(records, data_root, video_dir, video_root, processed_dir, manifest_name):
    output = manifest_path(data_root, manifest_name)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "instrument",
                "youtube_id",
                "sample_key",
                "video_path",
                "sample_dir",
            ],
        )
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["video_path"] = format_manifest_path(
                video_output_path(data_root, video_dir, record, video_root),
                data_root,
            )
            row["sample_dir"] = sample_output_dir(data_root, processed_dir, record).relative_to(data_root).as_posix()
            writer.writerow(row)
    return output


def read_manifest(data_root, manifest_name):
    output = manifest_path(data_root, manifest_name)
    with output.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_command_path(name):
    path = shutil.which(name)
    if path:
        return path
    candidate = Path(name).expanduser()
    if candidate.exists():
        return str(candidate)
    return None


def expand_paths(paths):
    return [Path(path).expanduser() for path in paths]


def detect_cookie_browser(extra_args):
    for index, arg in enumerate(extra_args):
        if arg == "--cookies-from-browser":
            if index + 1 >= len(extra_args):
                raise RuntimeError("`--cookies-from-browser` was provided without a browser name.")
            return extra_args[index + 1]
        if arg.startswith("--cookies-from-browser="):
            return arg.split("=", 1)[1]
    return None


def detect_cookie_file(extra_args):
    for index, arg in enumerate(extra_args):
        if arg == "--cookies":
            if index + 1 >= len(extra_args):
                raise RuntimeError("`--cookies` was provided without a cookie file path.")
            return extra_args[index + 1]
        if arg.startswith("--cookies="):
            return arg.split("=", 1)[1]
    return None


def has_yt_dlp_option(extra_args, option):
    return any(arg == option or arg.startswith(f"{option}=") for arg in extra_args)


def resolve_yt_dlp_js_runtime(preferred):
    if preferred in {"", "none", "off", "false"}:
        return None
    if preferred != "auto":
        return preferred
    for runtime in ("deno", "node", "bun", "qjs"):
        path = shutil.which(runtime)
        if path:
            if runtime == "qjs":
                return f"quickjs:{path}"
            return runtime
    return None


def build_yt_dlp_base_args(args, needs_js_runtime=False):
    base_args = []
    if needs_js_runtime and not has_yt_dlp_option(args.yt_dlp_extra_arg, "--js-runtimes"):
        js_runtime = resolve_yt_dlp_js_runtime(args.yt_dlp_js_runtime)
        if js_runtime:
            base_args.extend(["--js-runtimes", js_runtime])
        else:
            print(
                "[prepare_musices] warning: no JavaScript runtime found for yt-dlp. "
                "YouTube may return only image/storyboard formats. Install deno or node, "
                "or pass --yt-dlp-js-runtime manually.",
                file=sys.stderr,
            )
    return base_args + list(args.yt_dlp_extra_arg)


def validate_yt_dlp_cookie_source(extra_args):
    cookie_file = detect_cookie_file(extra_args)
    if cookie_file:
        cookie_path = Path(cookie_file).expanduser()
        if not cookie_path.exists():
            raise RuntimeError(
                f"Cookie file not found: {cookie_path}. "
                "Pass an existing Netscape-format cookies file, or remove the cookie option. "
                "For YouTube on Windows/WSL, the recommended path is to manually export a fresh "
                "Netscape-format `youtube_cookies.txt` from a private/incognito browser session "
                "and save it to that location. "
                "`bash tools/export_windows_edge_cookies.sh` is only a best-effort backup helper."
            )
        return

    browser_spec = detect_cookie_browser(extra_args)
    if not browser_spec:
        return

    browser_name = browser_spec.split(":", 1)[0].strip().lower()
    source = BROWSER_COOKIE_SOURCES.get(browser_name)
    if source is None:
        return

    binaries = source["binaries"]
    profile_dirs = expand_paths(source["dirs"])
    found_binary = next((name for name in binaries if resolve_command_path(name)), None)
    found_profile_dir = next((path for path in profile_dirs if path.exists()), None)
    if found_binary and found_profile_dir:
        return

    expected_dirs = ", ".join(f"`{path}`" for path in profile_dirs)
    binary_hint = ", ".join(f"`{name}`" for name in binaries)
    raise RuntimeError(
        "Unable to read browser cookies for yt-dlp. "
        f"Requested browser: `{browser_spec}`. "
        f"Expected a browser install ({binary_hint}) and a profile directory in {expected_dirs}, "
        f"but found binary={bool(found_binary)} and profile_dir={bool(found_profile_dir)}. "
        "This is usually not a filesystem permission problem. It usually means the browser is not "
        "installed in this Linux/WSL environment, has never been launched here, or you are trying "
        "to read cookies from a Windows browser while running inside WSL. "
        "Fix it by either: 1) installing that browser inside this environment and logging into "
        "YouTube once, then closing the browser; or 2) exporting cookies to a Netscape-format file "
        "and passing `--yt-dlp-extra-arg=--cookies --yt-dlp-extra-arg=/absolute/path/cookies.txt`."
    )


def resolve_yt_dlp_command(preferred):
    preferred_path = resolve_command_path(preferred)
    if preferred_path:
        return [preferred_path]
    if preferred != "yt-dlp":
        raise RuntimeError(
            f"Unable to resolve yt-dlp from `--yt-dlp-bin={preferred}`. "
            "Install it with `uv add \"yt-dlp[default]\" && uv sync`, "
            "or pass the correct executable path."
        )
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    raise RuntimeError(
        "yt-dlp is required for MUSICES downloads and size estimation. "
        "Install it with `uv add \"yt-dlp[default]\" && uv sync`."
    )


def resolve_ffmpeg_binary(preferred):
    preferred_path = resolve_command_path(preferred)
    if preferred_path:
        return preferred_path
    if preferred != "ffmpeg":
        raise RuntimeError(
            f"Unable to resolve ffmpeg from `--ffmpeg-bin={preferred}`. "
            "Pass a valid executable path, install system ffmpeg, or install "
            "`imageio-ffmpeg` with `uv add imageio-ffmpeg && uv sync`."
        )
    imageio_ffmpeg = require_imageio_ffmpeg()
    return imageio_ffmpeg.get_ffmpeg_exe()


def build_yt_dlp_command(
    yt_dlp_command,
    ffmpeg_binary=None,
    extra_args=None,
):
    command = list(yt_dlp_command)
    if ffmpeg_binary:
        command.extend(["--ffmpeg-location", ffmpeg_binary])
    if extra_args:
        command.extend(extra_args)
    return command


def extract_size_candidate(info):
    for key in ("filesize", "filesize_approx"):
        value = info.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def estimate_total_bytes(info):
    direct_size = extract_size_candidate(info)
    if direct_size is not None:
        return direct_size

    nested_total = 0
    nested_found = False
    for key in ("requested_downloads", "requested_formats"):
        entries = info.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_size = extract_size_candidate(entry)
            if entry_size is None:
                continue
            nested_total += entry_size
            nested_found = True
    if nested_found:
        return nested_total
    return None


def inspect_record_stats(record, yt_dlp_command, yt_dlp_extra_args):
    command = build_yt_dlp_command(
        yt_dlp_command,
        extra_args=[
            "--dump-single-json",
            "--quiet",
            "--no-warnings",
            "--skip-download",
            "-f",
            YT_DLP_FORMAT,
        ]
        + list(yt_dlp_extra_args)
        + [video_url(record)],
    )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or "").strip()
        return {
            "instrument": record["instrument"],
            "youtube_id": record["youtube_id"],
            "sample_key": record["sample_key"],
            "status": "error",
            "estimated_total_bytes": None,
            "url": video_url(record),
            "error_message": error_text or f"yt-dlp exited with code {result.returncode}",
        }

    payload = result.stdout.strip()
    if not payload:
        return {
            "instrument": record["instrument"],
            "youtube_id": record["youtube_id"],
            "sample_key": record["sample_key"],
            "status": "error",
            "estimated_total_bytes": None,
            "url": video_url(record),
            "error_message": "yt-dlp returned no metadata output",
        }

    try:
        info = json.loads(payload)
    except json.JSONDecodeError as exc:
        return {
            "instrument": record["instrument"],
            "youtube_id": record["youtube_id"],
            "sample_key": record["sample_key"],
            "status": "error",
            "estimated_total_bytes": None,
            "url": video_url(record),
            "error_message": f"failed to parse yt-dlp metadata: {exc}",
        }

    estimated_bytes = estimate_total_bytes(info)
    return {
        "instrument": record["instrument"],
        "youtube_id": record["youtube_id"],
        "sample_key": record["sample_key"],
        "status": "ok" if estimated_bytes is not None else "unknown_size",
        "estimated_total_bytes": estimated_bytes,
        "url": video_url(record),
        "error_message": "",
    }


def write_stats_files(records, args, yt_dlp_command):
    data_root = Path(args.data_root)
    rows = []
    by_instrument = {}
    estimated_total_bytes = 0
    estimated_record_count = 0
    unknown_record_count = 0

    for index, record in enumerate(records, start=1):
        print(f"[prepare_musices] stats {index}/{len(records)}: {record['sample_key']}")
        yt_dlp_args = build_yt_dlp_base_args(args, needs_js_runtime=True)
        row = inspect_record_stats(record, yt_dlp_command, yt_dlp_args)
        rows.append(row)

        instrument_summary = by_instrument.setdefault(
            record["instrument"],
            {
                "record_count": 0,
                "estimated_record_count": 0,
                "unknown_record_count": 0,
                "estimated_total_bytes": 0,
            },
        )
        instrument_summary["record_count"] += 1

        if row["estimated_total_bytes"] is None:
            unknown_record_count += 1
            instrument_summary["unknown_record_count"] += 1
        else:
            estimated_total_bytes += row["estimated_total_bytes"]
            estimated_record_count += 1
            instrument_summary["estimated_record_count"] += 1
            instrument_summary["estimated_total_bytes"] += row["estimated_total_bytes"]

    summary = {
        "record_count": len(records),
        "estimated_record_count": estimated_record_count,
        "unknown_record_count": unknown_record_count,
        "estimated_total_bytes": estimated_total_bytes,
        "estimated_total_human": format_bytes(estimated_total_bytes),
        "by_instrument": by_instrument,
        "records": rows,
    }

    json_output = stats_json_path(data_root, args.stats_json_name)
    csv_output = stats_csv_path(data_root, args.stats_csv_name)
    ensure_dir(json_output.parent)

    with json_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)

    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "instrument",
                "youtube_id",
                "sample_key",
                "status",
                "estimated_total_bytes",
                "url",
                "error_message",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return summary, json_output, csv_output


def load_stats_index(data_root, stats_json_name):
    json_path = stats_json_path(data_root, stats_json_name)
    if not json_path.exists():
        return None, {}
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    record_map = {}
    for row in payload.get("records", []):
        record_map[row["sample_key"]] = row.get("estimated_total_bytes")
    return payload.get("estimated_total_bytes"), record_map


def summarize_existing_downloads(records, data_root, video_dir, video_root=None):
    sizes = {}
    total_bytes = 0
    completed_count = 0
    for record in records:
        target = video_output_path(data_root, video_dir, record, video_root)
        size = target.stat().st_size if target.exists() else 0
        sizes[record["sample_key"]] = size
        if size > 0:
            completed_count += 1
            total_bytes += size
    return sizes, completed_count, total_bytes


def download_video(
    record,
    data_root,
    video_dir,
    yt_dlp_command,
    ffmpeg_binary,
    skip_existing,
    archive_path,
    yt_dlp_extra_args,
    video_root=None,
    estimated_total_bytes=None,
):
    target = video_output_path(data_root, video_dir, record, video_root)
    ensure_dir(target.parent)
    before_bytes = target.stat().st_size if target.exists() else 0
    if skip_existing and target.exists():
        return {
            "sample_key": record["sample_key"],
            "status": "skipped_existing",
            "output_path": str(target),
            "downloaded_bytes": before_bytes,
            "estimated_total_bytes": estimated_total_bytes,
            "error_message": "",
        }

    command = build_yt_dlp_command(
        yt_dlp_command,
        ffmpeg_binary=ffmpeg_binary,
        extra_args=[
            "-f",
            YT_DLP_FORMAT,
            "--merge-output-format",
            "mp4",
            "--continue",
            "--part",
            "--newline",
            "--progress",
            "--progress-template",
            DOWNLOAD_PROGRESS_TEMPLATE,
            "--download-archive",
            str(archive_path),
            "-o",
            str(target.with_suffix(".%(ext)s")),
        ]
        + list(yt_dlp_extra_args)
        + [video_url(record)],
    )

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        after_bytes = target.stat().st_size if target.exists() else before_bytes
        return {
            "sample_key": record["sample_key"],
            "status": "error",
            "output_path": str(target),
            "downloaded_bytes": after_bytes,
            "estimated_total_bytes": estimated_total_bytes,
            "error_message": f"yt-dlp exited with code {exc.returncode}",
        }

    after_bytes = target.stat().st_size if target.exists() else before_bytes
    if after_bytes > before_bytes and before_bytes > 0:
        status = "resumed"
    elif after_bytes > 0 and before_bytes == 0:
        status = "downloaded"
    else:
        status = "up_to_date"

    return {
        "sample_key": record["sample_key"],
        "status": status,
        "output_path": str(target),
        "downloaded_bytes": after_bytes,
        "estimated_total_bytes": estimated_total_bytes,
        "error_message": "",
    }


def write_download_failures(data_root, failures):
    output = download_failure_path(data_root)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_key",
                "status",
                "output_path",
                "downloaded_bytes",
                "estimated_total_bytes",
                "error_message",
            ],
        )
        writer.writeheader()
        for failure in failures:
            writer.writerow(failure)
    return output


def write_process_failures(data_root, failures):
    output = process_failure_path(data_root)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_key",
                "status",
                "video_path",
                "error_message",
            ],
        )
        writer.writeheader()
        for failure in failures:
            writer.writerow(failure)
    return output


def normalize_spectrogram(spectrogram, min_level_db):
    np = require_numpy()
    return np.clip((spectrogram - min_level_db) / -min_level_db, 0.0, 1.0)


def compute_mel_spectrogram(
    wav,
    sample_rate,
    fft_size,
    hop_size,
    num_mels,
    fmin,
    fmax,
    min_level_db,
    ref_level_db,
):
    np = require_numpy()
    librosa = require_librosa()
    stft = librosa.stft(y=wav, n_fft=fft_size, hop_length=hop_size, win_length=fft_size)
    magnitude = np.abs(stft)
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=fft_size,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel = np.dot(mel_basis, magnitude)
    min_level = np.exp(min_level_db / 20.0 * np.log(10))
    mel_db = 20.0 * np.log10(np.maximum(min_level, mel)) - ref_level_db
    return normalize_spectrogram(mel_db, min_level_db).T.astype(np.float32)


def align_waveform_length(wav, mel_frames, hop_size):
    np = require_numpy()
    target = mel_frames * hop_size
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)), mode="constant", constant_values=0.0)
    else:
        wav = wav[:target]
    return wav.astype(np.float32)


def fit_mel_frame_count(mel, target_frames):
    np = require_numpy()
    if target_frames is None or target_frames <= 0:
        return mel
    target_frames = int(target_frames)
    if mel.shape[0] > target_frames:
        return mel[:target_frames]
    if mel.shape[0] < target_frames:
        padding = np.zeros((target_frames - mel.shape[0], mel.shape[1]), dtype=mel.dtype)
        return np.concatenate([mel, padding], axis=0)
    return mel


def export_audio_and_mel(sample_dir, wav_path, args):
    np = require_numpy()
    librosa = require_librosa()
    wav, _ = librosa.load(str(wav_path), sr=args.sample_rate, mono=True)
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav)) * 0.999

    mel = compute_mel_spectrogram(
        wav=wav,
        sample_rate=args.sample_rate,
        fft_size=args.fft_size,
        hop_size=args.hop_size,
        num_mels=args.num_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        min_level_db=args.min_level_db,
        ref_level_db=args.ref_level_db,
    )
    mel = fit_mel_frame_count(mel, getattr(args, "clip_mel_frames", None))
    wav = align_waveform_length(wav, mel.shape[0], args.hop_size)

    np.save(sample_dir / "raw_audio.npy", wav.astype(np.float32), allow_pickle=False)
    np.save(sample_dir / "mel.npy", mel.astype(np.float32), allow_pickle=False)
    return mel.shape[0]


def audio_rms(wav_path, args):
    np = require_numpy()
    librosa = require_librosa()
    wav, _ = librosa.load(str(wav_path), sr=args.sample_rate, mono=True)
    if wav.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wav))))


def reset_directory(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)


def video_duration_and_fps(video_path):
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps is None or fps <= 0:
        fps = 25.0
    duration = frame_count / fps if frame_count and frame_count > 0 else 0.0
    return float(duration), float(fps)


def detect_video_segments(video_path, args, label=None):
    cv2 = require_cv2()
    np = require_numpy()
    duration, fps = video_duration_and_fps(video_path)
    start_sec = min(max(args.trim_start_sec, 0.0), duration)
    if duration - start_sec < args.min_segment_sec:
        return []
    if not args.shot_detection:
        return [(start_sec, duration - start_sec)]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    start_frame = int(round(start_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    previous_gray = None
    segment_start = start_sec
    segments = []
    frame_index = start_frame
    total_frames = max(0, int(round((duration - start_sec) * fps)))
    progress_label = label if label is not None else video_path.stem

    with make_progress(args, f"{progress_label} shot-detect", total=total_frames, unit="frame") as progress:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / fps
            if timestamp >= duration:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                diff = float(np.mean(np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32))))
                if diff >= args.shot_diff_threshold:
                    if timestamp - segment_start >= args.min_segment_sec:
                        segments.append((segment_start, timestamp - segment_start))
                    segment_start = timestamp
            previous_gray = gray
            frame_index += 1
            progress.update(1)

    cap.release()
    if duration - segment_start >= args.min_segment_sec:
        segments.append((segment_start, duration - segment_start))
    return segments


def black_frame_ratio(video_path, start_sec, duration_sec, args, sample_stride_frames=25, label=None):
    cv2 = require_cv2()
    np = require_numpy()
    _, fps = video_duration_and_fps(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    start_frame = int(round(start_sec * fps))
    end_frame = int(round((start_sec + duration_sec) * fps))
    sampled = 0
    black = 0
    stride = max(1, sample_stride_frames)
    progress_label = label if label is not None else video_path.stem
    frame_indices = list(range(start_frame, max(start_frame, end_frame), stride))

    with make_progress(args, f"{progress_label} black-check", total=len(frame_indices), unit="sample") as progress:
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                progress.update(1)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sampled += 1
            if float(np.mean(gray)) <= args.black_frame_threshold:
                black += 1
            progress.update(1)
            if frame_index >= end_frame:
                break

    cap.release()
    if sampled == 0:
        return 1.0
    return black / sampled


def normalize_flow_component(component, flow_clip):
    np = require_numpy()
    scaled = np.clip(component, -flow_clip, flow_clip)
    scaled = 127.0 + scaled * (127.0 / flow_clip)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def create_tvl1_flow_estimator(cv2):
    if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        return cv2.optflow.DualTVL1OpticalFlow_create()
    if hasattr(cv2, "DualTVL1OpticalFlow_create"):
        return cv2.DualTVL1OpticalFlow_create()
    raise RuntimeError(
        "TV-L1 optical flow requires opencv-contrib-python. "
        "Install/sync dependencies, then rerun the process stage. "
        "Use --flow-method farneback only for non-paper smoke tests."
    )


def compute_optical_flow(cv2, previous_gray, gray, flow_method, tvl1_estimator):
    if flow_method == "tvl1":
        return tvl1_estimator.calc(previous_gray, gray, None)
    return cv2.calcOpticalFlowFarneback(
        previous_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def extract_frames_and_flow(
    video_path,
    sample_dir,
    frame_size,
    frame_stride,
    flow_clip,
    flow_method,
    target_frame_count=None,
    start_sec=0.0,
    duration_sec=None,
    args=None,
    label=None,
):
    cv2 = require_cv2()
    np = require_numpy()
    image_dir = sample_dir / "image"
    flow_x_dir = sample_dir / "flow_x"
    flow_y_dir = sample_dir / "flow_y"
    for directory in [image_dir, flow_x_dir, flow_y_dir]:
        reset_directory(directory)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    start_frame = int(round(max(start_sec, 0.0) * fps))
    end_frame = None
    if duration_sec is not None:
        end_frame = int(round((max(start_sec, 0.0) + duration_sec) * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    frame_index = start_frame
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if end_frame is not None:
        total_read_frames = max(0, end_frame - start_frame)
    elif frame_count and frame_count > start_frame:
        total_read_frames = int(frame_count - start_frame)
    else:
        total_read_frames = None
    requested_frame_count = None
    selected_offsets = None
    if target_frame_count is not None and target_frame_count > 0:
        requested_frame_count = int(target_frame_count)
    if requested_frame_count is not None and total_read_frames:
        selected_frame_count = min(requested_frame_count, int(total_read_frames))
        selected_offsets = set(
            int(offset)
            for offset in np.linspace(0, total_read_frames - 1, num=selected_frame_count).round()
        )
    progress_args = args if args is not None else argparse.Namespace(progress=False)
    progress_label = label if label is not None else video_path.stem

    with make_progress(progress_args, f"{progress_label} read-frames", total=total_read_frames, unit="frame") as progress:
        while True:
            if end_frame is not None and frame_index >= end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            offset = frame_index - start_frame
            if selected_offsets is not None:
                should_keep = offset in selected_offsets
            else:
                should_keep = offset % max(frame_stride, 1) == 0
            if should_keep:
                frames.append(cv2.resize(frame, (frame_size, frame_size)))
            frame_index += 1
            progress.update(1)
    cap.release()

    if requested_frame_count is not None and len(frames) < requested_frame_count:
        raise RuntimeError(
            f"Not enough visual frames in clip: need {requested_frame_count}, "
            f"found {len(frames)} after reading {video_path}"
        )
    if len(frames) < 2:
        raise RuntimeError(f"Video segment too short to extract optical flow: {video_path}")

    previous_gray = None
    zero_flow = np.full((frame_size, frame_size), 127, dtype=np.uint8)
    tvl1_estimator = create_tvl1_flow_estimator(cv2) if flow_method == "tvl1" else None
    with make_progress(progress_args, f"{progress_label} {flow_method}-flow", total=len(frames), unit="frame") as progress:
        for frame_id, frame in enumerate(frames, start=1):
            cv2.imwrite(str(image_dir / f"{frame_id}.jpg"), frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is None:
                flow_x = zero_flow
                flow_y = zero_flow
            else:
                flow = compute_optical_flow(cv2, previous_gray, gray, flow_method, tvl1_estimator)
                flow_x = normalize_flow_component(flow[..., 0], flow_clip)
                flow_y = normalize_flow_component(flow[..., 1], flow_clip)
            cv2.imwrite(str(flow_x_dir / f"{frame_id}.jpg"), flow_x)
            cv2.imwrite(str(flow_y_dir / f"{frame_id}.jpg"), flow_y)
            previous_gray = gray
            progress.update(1)


def find_cluster(indices):
    min_idx = 0
    max_idx = -1
    min_num = indices[min_idx]
    max_num = indices[max_idx]
    for _ in range(len(indices) - 1):
        if (min_num + 5) not in indices:
            min_idx += 1
            min_num = indices[min_idx]
    for _ in range(len(indices) - 1):
        if (max_num - 5) not in indices:
            max_idx -= 1
            max_num = indices[max_idx]
    if min_num > max_num:
        max_num = min_num
    return min_num, max_num


def padding_square(image):
    cv2 = require_cv2()
    height, width = image.shape[:2]
    if height == width:
        return image
    larger_side = max(height, width)
    delta = abs(height - width)
    if larger_side == height:
        left, right = delta // 2, delta - delta // 2
        top = bottom = 0
    else:
        top, bottom = delta // 2, delta - delta // 2
        left = right = 0
    value = [127, 127, 127] if image.ndim == 3 else [127]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)


def compute_crop_bounds(sample_dir, args=None, label=None):
    cv2 = require_cv2()
    np = require_numpy()
    flow_x_paths = sorted((sample_dir / "flow_x").glob("*.jpg"), key=lambda path: int(path.stem))
    if not flow_x_paths:
        raise RuntimeError(f"No optical-flow frames found in {sample_dir}")

    sum_flow_x = None
    sum_flow_y = None
    progress_args = args if args is not None else argparse.Namespace(progress=False)
    progress_label = label if label is not None else sample_dir.name
    with make_progress(progress_args, f"{progress_label} crop-bounds", total=len(flow_x_paths), unit="frame") as progress:
        for flow_x_path in flow_x_paths:
            frame_id = flow_x_path.stem
            flow_y_path = sample_dir / "flow_y" / f"{frame_id}.jpg"
            flow_x = cv2.imread(str(flow_x_path), 0)
            flow_y = cv2.imread(str(flow_y_path), 0)
            if flow_x is None or flow_y is None:
                progress.update(1)
                continue
            diff_x = np.abs(flow_x.astype(np.int32) - 127)
            diff_y = np.abs(flow_y.astype(np.int32) - 127)
            sum_flow_x = diff_x if sum_flow_x is None else sum_flow_x + diff_x
            sum_flow_y = diff_y if sum_flow_y is None else sum_flow_y + diff_y
            progress.update(1)

    if sum_flow_x is None or sum_flow_y is None:
        return None

    total = sum_flow_x + sum_flow_y
    mask = (total > (2 * len(flow_x_paths))).astype(int)
    sum_w = np.where(np.sum(mask, axis=0) > 0)[0]
    sum_h = np.where(np.sum(mask, axis=1) > 0)[0]
    if len(sum_w) == 0 or len(sum_h) == 0:
        return None

    w_min, w_max = find_cluster(sum_w)
    h_min, h_max = find_cluster(sum_h)
    if (w_max - w_min) < 50 or (h_max - h_min) < 50:
        return None
    return int(w_min), int(w_max), int(h_min), int(h_max)


def crop_motion_region(sample_dir, args=None, label=None):
    cv2 = require_cv2()
    image_crop_dir = sample_dir / "image_crop"
    flow_x_crop_dir = sample_dir / "flow_x_crop"
    flow_y_crop_dir = sample_dir / "flow_y_crop"
    for directory in [image_crop_dir, flow_x_crop_dir, flow_y_crop_dir]:
        reset_directory(directory)

    progress_args = args if args is not None else argparse.Namespace(progress=False)
    progress_label = label if label is not None else sample_dir.name
    bounds = compute_crop_bounds(sample_dir, args=progress_args, label=progress_label)
    image_paths = sorted((sample_dir / "image").glob("*.jpg"), key=lambda path: int(path.stem))
    if not image_paths:
        raise RuntimeError(f"No extracted image frames found in {sample_dir}")

    with make_progress(progress_args, f"{progress_label} crop-write", total=len(image_paths), unit="frame") as progress:
        for image_path in image_paths:
            frame_id = image_path.stem
            image = cv2.imread(str(image_path))
            flow_x = cv2.imread(str(sample_dir / "flow_x" / f"{frame_id}.jpg"), 0)
            flow_y = cv2.imread(str(sample_dir / "flow_y" / f"{frame_id}.jpg"), 0)

            if bounds is not None:
                w_min, w_max, h_min, h_max = bounds
                image = image[h_min:h_max, w_min:w_max]
                flow_x = flow_x[h_min:h_max, w_min:w_max]
                flow_y = flow_y[h_min:h_max, w_min:w_max]

            image = padding_square(image)
            flow_x = padding_square(flow_x)
            flow_y = padding_square(flow_y)

            cv2.imwrite(str(image_crop_dir / f"{frame_id}.jpg"), image)
            cv2.imwrite(str(flow_x_crop_dir / f"{frame_id}.jpg"), flow_x)
            cv2.imwrite(str(flow_y_crop_dir / f"{frame_id}.jpg"), flow_y)
            progress.update(1)


def extract_audio_from_video(video_path, wav_path, ffmpeg_binary, skip_existing, start_sec=0.0, duration_sec=None):
    ensure_dir(wav_path.parent)
    if skip_existing and wav_path.exists():
        return wav_path
    command = [
        ffmpeg_binary,
        "-y",
        "-ss",
        str(max(start_sec, 0.0)),
    ]
    if duration_sec is not None:
        command.extend(["-t", str(duration_sec)])
    command.extend([
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ])
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path


def processed_sample_ready(sample_dir, args=None):
    required_files = [
        sample_dir / "raw_audio.npy",
        sample_dir / "mel.npy",
    ]
    required_dirs = [
        sample_dir / "image_crop",
        sample_dir / "flow_x_crop",
        sample_dir / "flow_y_crop",
    ]
    if not (all(path.exists() for path in required_files) and all(path.exists() for path in required_dirs)):
        return False
    if args is None:
        return True
    try:
        validate_av_sample(sample_dir, args)
    except BadAVSampleError:
        return False
    return True


def clean_direct_sample_payload(sample_dir):
    for name in [
        "source.wav",
        "raw_audio.npy",
        "mel.npy",
        "image",
        "flow_x",
        "flow_y",
        "image_crop",
        "flow_x_crop",
        "flow_y_crop",
    ]:
        path = sample_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def clean_legacy_flat_sample(sample_dir):
    clean_direct_sample_payload(sample_dir)
    if not sample_dir.exists():
        return
    for shot_dir in sample_dir.glob("shot_*"):
        if shot_dir.is_dir():
            clean_direct_sample_payload(shot_dir)


def process_clip(record, args, ffmpeg_binary, segment_id, clip_id, start_sec, duration_sec):
    data_root = Path(args.data_root)
    video_path = video_output_path(data_root, args.video_dir, record, args.video_root)
    sample_dir = clip_output_dir(data_root, args.processed_dir, record, segment_id, clip_id)
    label = (
        f"{record['sample_key']} shot{segment_id if segment_id is not None else 0} "
        f"clip{clip_id}"
    )

    if args.skip_existing and processed_sample_ready(sample_dir, args=args):
        return {
            "sample_dir": sample_dir,
            "mel_frames": None,
            "status": "skipped_existing",
        }

    ensure_dir(sample_dir)
    wav_path = sample_dir / "source.wav"
    extract_audio_from_video(
        video_path,
        wav_path,
        ffmpeg_binary,
        args.skip_existing,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )
    rms = audio_rms(wav_path, args)
    if rms < args.min_audio_rms:
        shutil.rmtree(sample_dir)
        return {
            "sample_dir": sample_dir,
            "mel_frames": None,
            "status": "skipped_silent",
        }

    black_ratio = black_frame_ratio(video_path, start_sec, duration_sec, args, label=label)
    if black_ratio >= args.max_black_ratio:
        shutil.rmtree(sample_dir)
        return {
            "sample_dir": sample_dir,
            "mel_frames": None,
            "status": "skipped_black",
        }

    extract_frames_and_flow(
        video_path=video_path,
        sample_dir=sample_dir,
        frame_size=args.frame_size,
        frame_stride=args.frame_stride,
        flow_clip=args.flow_clip,
        flow_method=args.flow_method,
        target_frame_count=args.visual_frame_count,
        start_sec=start_sec,
        duration_sec=duration_sec,
        args=args,
        label=label,
    )
    crop_motion_region(sample_dir, args=args, label=label)
    mel_frames = export_audio_and_mel(sample_dir, wav_path, args)
    return {
        "sample_dir": sample_dir,
        "mel_frames": mel_frames,
        "status": "processed",
    }


def iter_clip_windows(segments, args):
    clip_duration = float(args.clip_duration_sec)
    if clip_duration <= 0:
        raise RuntimeError(f"--clip-duration-sec must be positive, got {args.clip_duration_sec}")
    clip_hop = float(args.clip_hop_sec)
    if clip_hop <= 0:
        raise RuntimeError(f"--clip-hop-sec must be positive, got {args.clip_hop_sec}")
    if args.max_clips_per_segment is not None and args.max_clips_per_segment <= 0:
        raise RuntimeError(
            f"--max-clips-per-segment must be positive when set, got {args.max_clips_per_segment}"
        )

    for segment_index, (segment_start, segment_duration) in enumerate(segments):
        if segment_duration + 1e-6 < clip_duration:
            continue
        segment_end = segment_start + segment_duration
        clip_index = 0
        clip_start = segment_start
        while clip_start + clip_duration <= segment_end + 1e-6:
            yield segment_index, clip_index, clip_start, clip_duration, segment_start, segment_duration
            clip_index += 1
            if args.max_clips_per_segment is not None and clip_index >= args.max_clips_per_segment:
                break
            clip_start = segment_start + clip_index * clip_hop


def sample_clip_windows_for_video(record, args, clip_windows):
    max_clips = args.max_clips_per_video
    if max_clips is None:
        return clip_windows
    if max_clips <= 0:
        raise RuntimeError(f"--max-clips-per-video must be positive when set, got {max_clips}")
    if len(clip_windows) <= max_clips:
        return clip_windows

    rng = random.Random(f"{args.seed}:{record['sample_key']}")
    selected_indices = sorted(rng.sample(range(len(clip_windows)), max_clips))
    return [clip_windows[index] for index in selected_indices]


def process_record(record, args, ffmpeg_binary):
    data_root = Path(args.data_root)
    video_path = video_output_path(data_root, args.video_dir, record, args.video_root)
    root_sample_dir = sample_output_dir(data_root, args.processed_dir, record)
    segments = detect_video_segments(video_path, args, label=record["sample_key"])
    if not segments:
        return {"status": "no_segments", "processed": 0, "skipped": 0}

    if args.shot_detection:
        ensure_dir(root_sample_dir)
        clean_legacy_flat_sample(root_sample_dir)

    candidate_clip_windows = list(iter_clip_windows(segments, args))
    clip_windows = sample_clip_windows_for_video(record, args, candidate_clip_windows)
    if not clip_windows:
        return {"status": "no_clip_windows", "processed": 0, "skipped": 0}
    if len(clip_windows) < len(candidate_clip_windows):
        print(
            f"[prepare_musices] clip sampling: {record['sample_key']} "
            f"selected={len(clip_windows)} candidates={len(candidate_clip_windows)} "
            f"max_clips_per_video={args.max_clips_per_video}"
        )

    processed = 0
    skipped = 0
    for segment_index, clip_index, start_sec, duration_sec, segment_start, segment_duration in clip_windows:
        segment_id = segment_index if args.shot_detection else None
        try:
            result = process_clip(
                record,
                args,
                ffmpeg_binary,
                segment_id,
                clip_index,
                start_sec,
                duration_sec,
            )
        except RuntimeError as exc:
            skipped += 1
            print(
                f"[prepare_musices] clip skipped: {record['sample_key']} "
                f"segment={segment_index} clip={clip_index} "
                f"clip_start={start_sec:.2f}s clip_duration={duration_sec:.2f}s "
                f"segment_start={segment_start:.2f}s segment_duration={segment_duration:.2f}s | {exc}"
            )
            continue

        if result["status"] == "processed":
            processed += 1
            print(
                f"[prepare_musices] clip processed: {record['sample_key']} "
                f"segment={segment_index} clip={clip_index} "
                f"clip_start={start_sec:.2f}s clip_duration={duration_sec:.2f}s "
                f"mel_frames={result['mel_frames']}"
            )
        else:
            skipped += 1
            print(
                f"[prepare_musices] clip {result['status']}: {record['sample_key']} "
                f"segment={segment_index} clip={clip_index} "
                f"clip_start={start_sec:.2f}s clip_duration={duration_sec:.2f}s"
            )
    return {"status": "processed" if processed else "no_valid_segments", "processed": processed, "skipped": skipped}


def write_split_files(records, args):
    np = require_numpy()
    data_root = Path(args.data_root)
    processed_dir = Path(args.processed_dir)

    existing = []
    for record in records:
        sample_dir = sample_output_dir(data_root, processed_dir, record)
        mel_path = sample_dir / "mel.npy"
        audio_path = sample_dir / "raw_audio.npy"
        if not processed_sample_ready(sample_dir):
            continue
        mel = np.load(mel_path, mmap_mode="r")
        existing.append(
            {
                "sample_dir": sample_dir.relative_to(data_root).as_posix(),
                "mel_path": mel_path.relative_to(data_root).as_posix(),
                "audio_path": audio_path.relative_to(data_root).as_posix(),
                "mel_frames": int(mel.shape[0]),
            }
        )

    if not existing:
        raise RuntimeError("No processed samples found. Run the process stage first.")

    rng = random.Random(args.seed)
    existing.sort(key=lambda item: item["sample_dir"])
    rng.shuffle(existing)

    test_count = max(1, int(round(len(existing) * args.test_size)))
    test_items = existing[:test_count]
    train_items = existing[test_count:]
    if not train_items:
        raise RuntimeError("Test split consumed all samples. Reduce --test-size.")

    def write_lines(target_path, items):
        with target_path.open("w", encoding="utf-8") as handle:
            for item in sorted(items, key=lambda row: row["sample_dir"]):
                handle.write(
                    "|".join(
                        [
                            item["sample_dir"],
                            item["mel_path"],
                            item["audio_path"],
                            str(item["mel_frames"]),
                        ]
                    )
                    + "\n"
                )

    write_lines(Path(args.data_root) / args.train_split_name, train_items)
    write_lines(Path(args.data_root) / args.test_split_name, test_items)
    return len(train_items), len(test_items)


def run_download_stage(records, args, yt_dlp_command, ffmpeg_binary):
    data_root = Path(args.data_root)
    archive_path = download_archive_path(data_root, args.download_archive_name)
    ensure_dir(archive_path.parent)

    overall_estimated_bytes, estimated_bytes_by_key = load_stats_index(data_root, args.stats_json_name)
    current_sizes, completed_count, downloaded_total_bytes = summarize_existing_downloads(
        records,
        data_root,
        args.video_dir,
        args.video_root,
    )

    failures = []
    aborted = False
    yt_dlp_args = build_yt_dlp_base_args(args, needs_js_runtime=True)
    for index, record in enumerate(records, start=1):
        sample_key = record["sample_key"]
        estimated_total_bytes = estimated_bytes_by_key.get(sample_key)
        print(
            f"[prepare_musices] downloading {index}/{len(records)}: {sample_key} | "
            f"downloaded={format_bytes(downloaded_total_bytes)}"
            + (
                f" / estimated_total={format_bytes(overall_estimated_bytes)}"
                if overall_estimated_bytes is not None
                else ""
            )
        )
        before_bytes = current_sizes.get(sample_key, 0)
        result = download_video(
            record=record,
            data_root=data_root,
            video_dir=args.video_dir,
            yt_dlp_command=yt_dlp_command,
            ffmpeg_binary=ffmpeg_binary,
            skip_existing=args.skip_existing,
            archive_path=archive_path,
            yt_dlp_extra_args=yt_dlp_args,
            video_root=args.video_root,
            estimated_total_bytes=estimated_total_bytes,
        )

        after_bytes = result["downloaded_bytes"] or 0
        downloaded_total_bytes += max(after_bytes - before_bytes, 0)
        current_sizes[sample_key] = after_bytes
        if before_bytes == 0 and after_bytes > 0:
            completed_count += 1

        if result["status"] == "error":
            failures.append(result)
            print(
                f"[prepare_musices] download failed: {sample_key} | "
                f"error={result['error_message']}"
            )
            if args.abort_on_download_error:
                aborted = True
                break
            continue

        print(
            f"[prepare_musices] download {result['status']}: {sample_key} | "
            f"file={format_bytes(after_bytes)} | cumulative={format_bytes(downloaded_total_bytes)}"
        )

    failure_output = write_download_failures(data_root, failures)
    print(
        f"[prepare_musices] download summary: completed={completed_count}/{len(records)}, "
        f"failures={len(failures)}, failure_log={failure_output}"
    )
    if aborted:
        raise RuntimeError("Aborted because --abort-on-download-error was enabled.")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    ensure_dir(data_root)
    records = load_records(args.json_path, max_videos=args.max_videos)
    if args.action in {"stats", "download", "all"}:
        validate_yt_dlp_cookie_source(args.yt_dlp_extra_arg)

    if args.action in {"manifest", "all"}:
        output = write_manifest(
            records=records,
            data_root=data_root,
            video_dir=args.video_dir,
            video_root=args.video_root,
            processed_dir=args.processed_dir,
            manifest_name=args.manifest_name,
        )
        print(f"[prepare_musices] wrote manifest: {output}")
        if args.action == "manifest":
            return

    if args.action in {"stats", "all"}:
        yt_dlp_command = resolve_yt_dlp_command(args.yt_dlp_bin)
        summary, json_output, csv_output = write_stats_files(records, args, yt_dlp_command)
        print(
            f"[prepare_musices] wrote stats: {json_output}, {csv_output} | "
            f"estimated={format_bytes(summary['estimated_total_bytes'])}, "
            f"unknown={summary['unknown_record_count']}"
        )
        if args.action == "stats":
            return

    if args.action in {"download", "all"}:
        if args.skip_download:
            print("[prepare_musices] skipping download stage")
        else:
            yt_dlp_command = resolve_yt_dlp_command(args.yt_dlp_bin)
            ffmpeg_binary = resolve_ffmpeg_binary(args.ffmpeg_bin)
            run_download_stage(records, args, yt_dlp_command, ffmpeg_binary)
        if args.action == "download":
            return

    if args.action in {"process", "all"}:
        cv2 = require_cv2()
        ensure_cv2_video_backend(cv2)
        ffmpeg_binary = resolve_ffmpeg_binary(args.ffmpeg_bin)
        process_failures = []
        for index, record in enumerate(records, start=1):
            video_path = video_output_path(data_root, args.video_dir, record, args.video_root)
            if not video_path.exists():
                print(f"[prepare_musices] skip missing video {record['sample_key']}: {video_path}")
                process_failures.append(
                    {
                        "sample_key": record["sample_key"],
                        "status": "missing_video",
                        "video_path": str(video_path),
                        "error_message": "video file does not exist",
                    }
                )
                continue
            print(f"[prepare_musices] processing {index}/{len(records)}: {record['sample_key']}")
            try:
                result = process_record(record, args, ffmpeg_binary)
            except RuntimeError as exc:
                status = "unreadable_video" if "Unable to open video" in str(exc) else "process_error"
                print(
                    f"[prepare_musices] skip unreadable/invalid video {record['sample_key']}: "
                    f"{video_path} | {exc}"
                )
                process_failures.append(
                    {
                        "sample_key": record["sample_key"],
                        "status": status,
                        "video_path": str(video_path),
                        "error_message": str(exc),
                    }
                )
                continue
            print(
                f"[prepare_musices] process summary: {record['sample_key']} | "
                f"status={result['status']} processed={result['processed']} skipped={result['skipped']}"
            )
        failure_output = write_process_failures(data_root, process_failures)
        print(
            f"[prepare_musices] process failure summary: "
            f"failures={len(process_failures)}, failure_log={failure_output}"
        )
        if args.action == "process":
            return

    if args.action == "splits":
        print(
            "[prepare_musices] warning: the splits action is kept for legacy compatibility. "
            "Use `python main.py split-data -- --data-root ...` for the paper-style "
            "train/val/test split protocol."
        )
        train_count, test_count = write_split_files(records, args)
        print(
            f"[prepare_musices] wrote split files: "
            f"{args.train_split_name} ({train_count}), {args.test_split_name} ({test_count})"
        )
    elif args.action == "all":
        print("[prepare_musices] processing complete. Run `python main.py split-data -- --data-root ...` next.")


if __name__ == "__main__":
    main()
