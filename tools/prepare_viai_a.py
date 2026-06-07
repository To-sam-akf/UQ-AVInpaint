import argparse
from pathlib import Path

from tools import prepare_musices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare VIAI-A audio-only samples from MUSICES raw videos."
    )
    parser.add_argument("--json", dest="json_path", default="/root/shared-nvme/data/MUSICES.json")
    parser.add_argument("--data-root", default="/root/shared-nvme/data")
    parser.add_argument("--video-dir", default="raw_videos")
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--fft-size", type=int, default=1280)
    parser.add_argument("--hop-size", type=int, default=320)
    parser.add_argument("--num-mels", type=int, default=80)
    parser.add_argument("--fmin", type=float, default=125.0)
    parser.add_argument("--fmax", type=float, default=7600.0)
    parser.add_argument("--min-level-db", type=float, default=-100.0)
    parser.add_argument("--ref-level-db", type=float, default=20.0)
    parser.add_argument(
        "--trim-start-sec",
        type=float,
        default=0.0,
        help="Trim this many seconds from the start before extracting audio.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Optional duration to extract. Leave unset to use the rest of the video.",
    )
    parser.add_argument(
        "--min-mel-frames",
        type=int,
        default=200,
        help="Skip samples shorter than the 4-second VIAI-A training window.",
    )
    return parser.parse_args()


def audio_sample_ready(sample_dir):
    return (sample_dir / "raw_audio.npy").exists() and (sample_dir / "mel.npy").exists()


def process_record(record, args, ffmpeg_binary):
    data_root = Path(args.data_root)
    video_path = prepare_musices.video_output_path(
        data_root,
        args.video_dir,
        record,
        args.video_root,
    )
    if not video_path.exists():
        print(f"[prepare_viai_a] skip missing video {record['sample_key']}: {video_path}")
        return "missing"

    sample_dir = prepare_musices.sample_output_dir(data_root, args.processed_dir, record)
    if args.skip_existing and audio_sample_ready(sample_dir):
        print(f"[prepare_viai_a] skipped existing: {record['sample_key']} -> {sample_dir}")
        return "skipped_existing"

    prepare_musices.ensure_dir(sample_dir)
    wav_path = sample_dir / "source.wav"
    # 从视频里抽音频，默认抽取 trim_start_sec 之后的剩余整段
    prepare_musices.extract_audio_from_video(
        video_path,
        wav_path,
        ffmpeg_binary,
        skip_existing=args.skip_existing,
        start_sec=args.trim_start_sec,
        duration_sec=args.duration_sec,
    )
    mel_frames = prepare_musices.export_audio_and_mel(sample_dir, wav_path, args)
    if mel_frames < args.min_mel_frames:
        print(
            f"[prepare_viai_a] skipped short audio: {record['sample_key']} "
            f"mel_frames={mel_frames} required={args.min_mel_frames}"
        )
        return "skipped_short"

    print(
        f"[prepare_viai_a] processed: {record['sample_key']} "
        f"mel_frames={mel_frames} -> {sample_dir}"
    )
    return "processed"


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    prepare_musices.ensure_dir(data_root)
    records = prepare_musices.load_records(args.json_path, max_videos=args.max_videos)
    ffmpeg_binary = prepare_musices.resolve_ffmpeg_binary(args.ffmpeg_bin)

    counts = {}
    for index, record in enumerate(records, start=1):
        print(f"[prepare_viai_a] processing {index}/{len(records)}: {record['sample_key']}")
        status = process_record(record, args, ffmpeg_binary)
        counts[status] = counts.get(status, 0) + 1

    summary = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    print(f"[prepare_viai_a] summary: {summary}")


if __name__ == "__main__":
    main()
