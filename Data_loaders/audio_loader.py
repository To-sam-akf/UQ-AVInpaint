import os
from os.path import dirname, join, expanduser
import sys
sys.path.append('..')
import torch
import random
from torch.utils.data.sampler import Sampler
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from torch.utils import data as data_utils
import glob
from sklearn.model_selection import train_test_split
import cv2
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input
import numpy as np
import Options_inpainting
from utils.av_sample_validation import (
    BadAVSampleError,
    av_window_requirements,
    inspect_av_sample,
    log_bad_sample,
    validate_av_sample,
)

hparams = Options_inpainting.Inpainting_Config()

fs = hparams.sample_rate


def get_hop_size():
    return hparams.hop_size


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


def to_categorical(values, num_classes):
    values = np.asarray(values, dtype=np.int64)
    categorical = np.zeros((values.size, num_classes), dtype=np.float32)
    categorical[np.arange(values.size), values] = 1.0
    return categorical


def ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)


def assert_ready_for_upsampling(x, c):
    assert len(x) % len(c) == 0 and len(x) // len(c) == get_hop_size()


def _bad_sample_log_path():
    return getattr(hparams, "bad_sample_log", None)


def _handle_bad_av_sample(error, source, phase=None, split_name=None, sample_path=None):
    if getattr(hparams, "strict_av_samples", False):
        raise error
    sample_path = sample_path or getattr(error, "sample_path", "")
    log_path = log_bad_sample(
        hparams.data_root,
        _bad_sample_log_path(),
        source=source,
        phase=phase,
        split_name=split_name,
        sample_path=sample_path,
        error=error,
    )
    print(
        f"[VIAI-AV data] skipped bad sample: {sample_path} "
        f"reason={getattr(error, 'reason', error.__class__.__name__)} log={log_path}"
    )
    return None


def _alignment_error(reason, path, message, found_mel_frames=None, found_audio_steps=None):
    requirements = av_window_requirements(hparams)
    record = {
        "sample_path": str(path),
        "reason": reason,
        "required_video_frames": requirements["required_video_frames"],
        "found_image_frames": "",
        "found_flow_x_frames": "",
        "found_flow_y_frames": "",
        "required_mel_frames": requirements["required_mel_frames"],
        "found_mel_frames": found_mel_frames,
        "required_audio_steps": requirements["required_audio_steps"],
        "found_audio_steps": found_audio_steps,
        "error_message": message,
    }
    return BadAVSampleError(reason, path, record=record, error_message=message)


class _NPYDataSource(FileDataSource):
    def __init__(self, data_root, col, speaker_id=None,
                 train=True, test_size=0.05, test_num_samples=None, random_state=1234,
                 phase=None, split_name=None):
        self.data_root = data_root
        self.col = col
        self.lengths = []
        self.speaker_id = speaker_id
        self.multi_speaker = False
        self.speaker_ids = None
        self.train = train
        self.test_size = test_size
        self.test_num_samples = test_num_samples
        self.random_state = random_state
        self.phase = phase if phase is not None else ("train" if train else "test")
        self.split_name = split_name


    def collect_files(self):
        metadata = self.split_name if self.split_name is not None else self.phase + hparams.new_split_name
        meta = join(self.data_root, metadata)
        with open(meta, "rb") as f:
            lines = [line for line in f.readlines() if line.strip()]
        if not lines:
            self.lengths = []
            return []
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4 or len(l) == 5
        self.multi_speaker = len(l) == 5
        self.lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[-1]) * 1280, lines))

        paths_relative = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), paths_relative))

        if self.multi_speaker:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-2]), lines))
            self.speaker_ids = speaker_ids
            if self.speaker_id is not None:
                # Filter by speaker_id
                # using multi-speaker dataset as a single speaker dataset
                indices = np.array(speaker_ids) == self.speaker_id
                paths = list(np.array(paths)[indices])

                # Filter by train/tset
                self.lengths = list(np.array(self.lengths)[indices])

                # aha, need to cast numpy.int64 to int
                self.lengths = list(map(int, self.lengths))
                self.multi_speaker = False

                return paths

        # Filter by train/test
        paths = list(np.array(paths))
        lengths_np = list(np.array(self.lengths))
        self.lengths = list(map(int, lengths_np))

        if self.multi_speaker:
            speaker_ids_np = list(np.array(self.speaker_ids))
            self.speaker_ids = list(map(int, speaker_ids_np))
            assert len(paths) == len(self.speaker_ids)
        paths = sorted(paths)
        return paths

    def collect_features(self, path):
        return np.load(path)


class ImageDataSource(FileDataSource):
    def __init__(self, data_root, col, speaker_id=None,
                 train=True, test_size=0.05, test_num_samples=None, random_state=1234,
                 phase=None, split_name=None):
        self.data_root = data_root
        self.col = col
        self.lengths = []
        self.speaker_id = speaker_id
        self.multi_speaker = False
        self.speaker_ids = None
        self.train = train
        self.test_size = test_size
        self.test_num_samples = test_num_samples
        self.random_state = random_state
        self.phase = phase if phase is not None else ("train" if train else "test")
        self.split_name = split_name

    def collect_files(self):
        metadata = self.split_name if self.split_name is not None else self.phase + hparams.new_split_name
        meta = join(self.data_root, metadata)
        with open(meta, "rb") as f:
            lines = [line for line in f.readlines() if line.strip()]
        if not lines:
            self.lengths = []
            return []
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4 or len(l) == 5
        self.multi_speaker = len(l) == 5
        self.lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))

        paths_relative = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), paths_relative))

        # Filter by train/test
        paths = list(np.array(paths))
        lengths_np = list(np.array(self.lengths))
        self.lengths = list(map(int, lengths_np))

        paths = sorted(paths)
        return paths

    def collect_features(self, path):
        video_block, flow_block, start = sample_data_new(path, self.train, hparams=hparams)
        return video_block, flow_block, start, path


def sample_data_new(data_path, train=True, hparams=hparams):
    validation_record = validate_av_sample(data_path, hparams)
    flow_x_crop_path = os.path.join(data_path, 'flow_x_crop')
    flow_y_crop_path = os.path.join(data_path, 'flow_y_crop')
    image_crop_path = os.path.join(data_path, 'image_crop')
    max_time_steps = hparams.max_time_steps
    num_images = min(
        int(validation_record["found_image_frames"]),
        int(validation_record["found_flow_x_frames"]),
        int(validation_record["found_flow_y_frames"]),
    )
    max_time_second = max_time_steps / hparams.sample_rate

    frame_stride = max(1, int(hparams.image_hope_size))
    visual_frame_count = int(getattr(hparams, "visual_frame_count", 0))
    use_image_num = visual_frame_count if visual_frame_count > 0 else int(np.floor(max_time_second / (0.04 * frame_stride)))
    if use_image_num <= 0:
        raise ValueError("use_image_num must be positive; check max_time_steps, visual_frame_count, and image_hope_size")

    last_offset = (use_image_num - 1) * frame_stride
    visual_frame_interval_sec = float(
        getattr(hparams, "visual_frame_interval_sec", 0.04 * frame_stride)
    )
    mel_frames_per_visual_frame = visual_frame_interval_sec * hparams.sample_rate / hparams.hop_size
    mel_window_frames = int(round(use_image_num * mel_frames_per_visual_frame))
    max_start_by_mel = None
    if validation_record["found_mel_frames"] is not None:
        mel_frames = int(validation_record["found_mel_frames"])
        max_start_by_mel = int(
            np.floor((mel_frames - mel_window_frames) / mel_frames_per_visual_frame)
        )
    min_start = 25
    max_start = num_images - 1 - last_offset - 25
    if max_start_by_mel is not None:
        max_start = min(max_start, max_start_by_mel)
    if max_start < min_start:
        min_start = 0
        max_start = num_images - 1 - last_offset
        if max_start_by_mel is not None:
            max_start = min(max_start, max_start_by_mel)
    if max_start < min_start:
        raise BadAVSampleError(
            "no_aligned_av_window",
            data_path,
            record=validation_record,
            error_message=(
                f"Not enough aligned audio/video frames in {data_path}: "
                f"need video_frames={last_offset + 1}, mel_frames={mel_window_frames}; "
                f"found video_frames={num_images}"
            ),
        )
    start_candidates = np.arange(min_start, max_start + 1)
    image_start = int(np.random.choice(start_candidates))

    # assert hparams.load_num > 0
    start = []
    start.append(image_start)
    for ln in range(1, hparams.load_num):
        separated = [candidate for candidate in start_candidates if all(abs(candidate - used) > 10 for used in start)]
        candidates = separated if separated else start_candidates
        start.append(int(np.random.choice(candidates)))
    image_rescal_size = hparams.image_rescal_size
    image_size = hparams.image_size
    if train:
        video_block = np.zeros(
                (hparams.load_num, use_image_num, hparams.image_rescal_size, hparams.image_rescal_size, 3))
        flow_block = np.zeros(
                (hparams.load_num, use_image_num, hparams.image_rescal_size, hparams.image_rescal_size, 2))
        crop_x = np.random.randint(0, image_rescal_size - image_size) if image_rescal_size > image_size else 0
        crop_y = np.random.randint(0, image_rescal_size - image_size) if image_rescal_size > image_size else 0
        flip = np.random.randint(0, 2)
    else:
        video_block = np.zeros(
                (hparams.load_num, use_image_num, hparams.image_size, hparams.image_size, 3))
        flow_block = np.zeros(
                (hparams.load_num, use_image_num, hparams.image_size, hparams.image_size, 2))
        crop_x = 0
        crop_y = 0
    if hparams.image or hparams.flow:
        for ln in range(hparams.load_num):
            i = 0
            for item in range(start[ln], start[ln] + use_image_num * frame_stride, frame_stride):
                flow_x_path = os.path.join(flow_x_crop_path, str(item + 1) + '.jpg')
                flow_y_path = os.path.join(flow_y_crop_path, str(item + 1) + '.jpg')
                image_path = os.path.join(image_crop_path, str(item + 1) + '.jpg')
                if hparams.image:
                    image = cv2.imread(image_path)
                    if image is None:
                        raise BadAVSampleError(
                            "unreadable_image_frame",
                            data_path,
                            record=validation_record,
                            error_message=f"Unable to read image frame: {image_path}",
                        )
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if train:
                        image = cv2.resize(image, (image_rescal_size, image_rescal_size))
                        if flip:
                            image = np.fliplr(image)
                    else:
                        image = cv2.resize(image, (image_size, image_size))
                    image = (image - 127.) / 128.
                    video_block[ln, i, :] = image
                if hparams.flow:
                    flow_x = cv2.imread(flow_x_path, 0)
                    flow_y = cv2.imread(flow_y_path, 0)
                    if flow_x is None or flow_y is None:
                        raise BadAVSampleError(
                            "unreadable_flow_frame",
                            data_path,
                            record=validation_record,
                            error_message=(
                                f"Unable to read flow frame: flow_x={flow_x_path}, "
                                f"flow_y={flow_y_path}"
                            ),
                        )
                    if train:
                        flow_x = cv2.resize(flow_x, (image_rescal_size, image_rescal_size))
                        flow_y = cv2.resize(flow_y, (image_rescal_size, image_rescal_size))

                        if flip:
                            flow_x = np.fliplr(flow_x)
                            flow_y = np.fliplr(flow_y)
                    else:
                        flow_x = cv2.resize(flow_x, (image_size, image_size))
                        flow_y = cv2.resize(flow_y, (image_size, image_size))
                    flow_y = (flow_y - 127.) / 128.
                    flow_x = (flow_x - 127.) / 128.

                    flow_block[ln, i, :, :, 0] = flow_x
                    flow_block[ln, i, :, :, 1] = flow_y
                i += 1
    video_block = video_block[:, :, crop_x:crop_x + image_size,
                  crop_y:crop_y + image_size]
    flow_block = flow_block[:, :, crop_x:crop_x + image_size,
                 crop_y:crop_y + image_size]
    video_block = video_block.transpose((0, 1, 4, 2, 3))
    flow_block = flow_block.transpose((0, 1, 4, 2, 3))
    return video_block, flow_block, start


def load_image(path, train, hparams=hparams):
    flow_x_crop_path = os.path.join(path, 'flow_x_crop')
    flow_y_crop_path = os.path.join(path, 'flow_y_crop')
    image_crop_path = os.path.join(path, 'image_crop')
    find_all_flows = glob.glob(os.path.join(flow_x_crop_path, '*.jpg'))
    len_flows = len(find_all_flows)
    image_rescal_size = hparams.image_rescal_size
    image_size = hparams.image_size
    if train:
        video_block = np.zeros(
                (len_flows, hparams.image_rescal_size, hparams.image_rescal_size, 3))
        flow_block = np.zeros(
                (len_flows,  hparams.image_rescal_size, hparams.image_rescal_size, 2))
        crop_x = np.random.randint(0, image_rescal_size - image_size) if image_rescal_size > image_size else 0
        crop_y = np.random.randint(0, image_rescal_size - image_size) if image_rescal_size > image_size else 0
        flip = np.random.randint(0, 2)
    else:
        video_block = np.zeros(
                (hparams.load_num, hparams.image_size, hparams.image_size, 3))
        flow_block = np.zeros(
                (hparams.load_num, hparams.image_size, hparams.image_size, 2))
        crop_x = 0
        crop_y = 0
    i = 0
    for item in range(len_flows):
        flow_x_path = os.path.join(flow_x_crop_path, str(item + 1) + '.jpg')
        flow_y_path = os.path.join(flow_y_crop_path, str(item + 1) + '.jpg')
        image_path = os.path.join(image_crop_path, str(item + 1) + '.jpg')
        if hparams.image:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if train:
                image = cv2.resize(image, (image_rescal_size, image_rescal_size))
                if flip:
                    image = np.fliplr(image)
            else:
                image = cv2.resize(image, (image_size, image_size))
            image = (image - 127.) / 128.
            video_block[i] = image
        if hparams.flow:
            flow_x = cv2.imread(flow_x_path, 0)
            flow_y = cv2.imread(flow_y_path, 0)
            if train:
                flow_x = cv2.resize(flow_x, (image_rescal_size, image_rescal_size))
                flow_y = cv2.resize(flow_y, (image_rescal_size, image_rescal_size))

                if flip:
                    flow_x = np.fliplr(flow_x)
                    flow_y = np.fliplr(flow_y)
            else:
                flow_x = cv2.resize(flow_x, (image_size, image_size))
                flow_y = cv2.resize(flow_y, (image_size, image_size))
            flow_y = (flow_y - 127.) / 128.
            flow_x = (flow_x - 127.) / 128.

            flow_block[i, :, :, 0] = flow_x
            flow_block[i, :, :, 1] = flow_y
        i += 1
    video_block = video_block[:, crop_x:crop_x + image_size,
                  crop_y:crop_y + image_size,:]
    flow_block = flow_block[:, crop_x:crop_x + image_size,
                 crop_y:crop_y + image_size,:]
    video_block = video_block.transpose((0, 3, 1, 2))
    flow_block = flow_block.transpose((0, 3, 1, 2))
    return video_block, flow_block


class RawAudioDataSource(_NPYDataSource):
    def __init__(self, data_root, **kwargs):
        super(RawAudioDataSource, self).__init__(data_root, 2, **kwargs)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, **kwargs):
        super(MelSpecDataSource, self).__init__(data_root, 1, **kwargs)


class ImageSpecDataSource(ImageDataSource):
    def __init__(self, data_root, **kwargs):
        super(ImageSpecDataSource, self).__init__(data_root, 0, **kwargs)


class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    """Partially randomized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batches
    """

    def __init__(self, lengths, batch_size=16, batch_group_size=None,
                 permutate=True):
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths))

        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].view(-1, self.batch_size)[perm, :].view(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)


class PyTorchDataset(object):
    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel
        # alias
        self.multi_speaker = X.file_data_source.multi_speaker

    def __getitem__(self, idx):
        if self.Mel is None:
            mel = None
        else:
            mel = self.Mel[idx]

        raw_audio = self.X[idx]
        if self.multi_speaker:
            speaker_id = self.X.file_data_source.speaker_ids[idx]
        else:
            speaker_id = None

        # (x,c,g)
        return raw_audio, mel, speaker_id

    def __len__(self):
        return len(self.X)


class PyTorchImageDataset(object):
    def __init__(self, X, Mel, Image):
        self.X = X
        self.Mel = Mel
        self.Image = Image
        self.phase = getattr(Image.file_data_source, "phase", "")
        self.split_name = getattr(Image.file_data_source, "split_name", "")
        # alias
        self.multi_speaker = X.file_data_source.multi_speaker

    def __getitem__(self, idx):
        try:
            if self.Mel is None:
                mel = None
            else:
                mel = self.Mel[idx]

            raw_audio = self.X[idx]
            video_block, flow_block, start, path = self.Image[idx]
            if self.multi_speaker:
                speaker_id = self.X.file_data_source.speaker_ids[idx]
            else:
                speaker_id = None
        except BadAVSampleError as exc:
            return _handle_bad_av_sample(
                exc,
                source="dataloader",
                phase=self.phase,
                split_name=self.split_name,
                sample_path=getattr(exc, "sample_path", ""),
            )

        # (x,c,g)
        return raw_audio, mel, video_block, flow_block, start, speaker_id, path, self.phase, self.split_name

    def __len__(self):
        return len(self.X)


def collate_fn(batch):
    """Create batch

    Args:
        batch(tuple): List of tuples
            - x[0] (ndarray,int) : list of (T,)
            - x[1] (ndarray,int) : list of (T, D)
            - x[2] (ndarray,int) : list of (1,), speaker id
    Returns:
        tuple: Tuple of batch
            - x (FloatTensor) : Network inputs (B, C, T)
            - y (LongTensor)  : Network targets (B, T, 1)
    """

    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    local_conditioning = len(batch[0]) >= 2 and hparams.cin_channels > 0
    global_conditioning = len(batch[0]) >= 3 and hparams.file_channel > 0

    if hparams.max_time_sec is not None:
        max_time_steps = int(hparams.max_time_sec * hparams.sample_rate)
    elif hparams.max_time_steps is not None:
        max_time_steps = hparams.max_time_steps
    else:
        max_time_steps = None
    max_time_second = max_time_steps / hparams.sample_rate

    frame_stride = max(1, int(hparams.image_hope_size))
    visual_frame_count = int(getattr(hparams, "visual_frame_count", 0))
    use_image_num = visual_frame_count if visual_frame_count > 0 else int(np.floor(max_time_second / (0.04 * frame_stride)))
    visual_frame_interval_sec = float(
        getattr(hparams, "visual_frame_interval_sec", 0.04 * frame_stride)
    )
    mel_frames_per_visual_frame = visual_frame_interval_sec * hparams.sample_rate / hparams.hop_size
    mel_window_frames = int(round(use_image_num * mel_frames_per_visual_frame))
    audio_window_steps = mel_window_frames * hparams.hop_size
    # Time resolution adjustment
    video_block = []
    flow_block = []
    if local_conditioning:
        new_batch = []
        for idx in range(len(batch)):
            x, c, video, flow, start, g, path, phase, split_name = batch[idx]
            try:
                if hparams.upsample_conditional_features:
                    if len(c) == 0 or len(x) % len(c) != 0 or len(x) // len(c) != get_hop_size():
                        raise _alignment_error(
                            "upsampling_mismatch",
                            path,
                            (
                                f"Sample is not ready for upsampling: {path}, "
                                f"audio_steps={len(x)}, mel_frames={len(c)}, hop_size={get_hop_size()}"
                            ),
                            found_mel_frames=len(c),
                            found_audio_steps=len(x),
                        )
                    if max_time_steps is not None:
                        max_steps = ensure_divisible(max_time_steps, get_hop_size(), True)
                        if len(x) < max_steps:
                            raise _alignment_error(
                                "insufficient_audio_steps",
                                path,
                                (
                                    f"Sample is shorter than the configured audio window: {path}, "
                                    f"audio_steps={len(x)}, required={max_steps}"
                                ),
                                found_mel_frames=len(c),
                                found_audio_steps=len(x),
                            )

                        for ln in range(hparams.load_num):
                            mel_start = int(round(start[ln] * mel_frames_per_visual_frame))
                            mel_end = mel_start + mel_window_frames
                            audio_start = mel_start * hparams.hop_size
                            audio_end = audio_start + audio_window_steps
                            if mel_end > len(c) or audio_end > len(x):
                                raise _alignment_error(
                                    "alignment_exceeds_sample_length",
                                    path,
                                    (
                                        f"Video/audio alignment exceeds sample length: {path}, "
                                        f"mel={mel_start}:{mel_end}/{len(c)}, "
                                        f"audio={audio_start}:{audio_end}/{len(x)}"
                                    ),
                                    found_mel_frames=len(c),
                                    found_audio_steps=len(x),
                                )
                            c1 = c[mel_start:mel_end]
                            x1 = x[audio_start:audio_end]
                            new_batch.append((x1, c1, g, os.path.join(path, str(start[ln]))))
                        video_block.append(torch.FloatTensor(video))
                        flow_block.append(torch.FloatTensor(flow))
            except BadAVSampleError as exc:
                _handle_bad_av_sample(
                    exc,
                    source="collate",
                    phase=phase,
                    split_name=split_name,
                    sample_path=path,
                )
        batch = new_batch
    if not batch:
        return None

    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # (B, T, C)
    # pad for time-axis
    if is_mulaw_quantize(hparams.input_type):
        x_batch = np.array([
            _pad_2d(to_categorical(x[0], hparams.quantize_channels), max_input_len)
            for x in batch
        ], dtype=np.float32)
    else:
        x_batch = np.array([_pad_2d(x[0].reshape(-1, 1), max_input_len)
                            for x in batch], dtype=np.float32)
    assert len(x_batch.shape) == 3

    # (B, T)
    if is_mulaw_quantize(hparams.input_type):
        y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    else:
        y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.float32)
    assert len(y_batch.shape) == 2

    # (B, T, D)
    if local_conditioning:
        max_len = max([len(x[1]) for x in batch])
        c_batch = np.array([_pad_2d(x[1], max_len) for x in batch], dtype=np.float32)
        assert len(c_batch.shape) == 3
        # (B x C x T)
        c_batch = torch.FloatTensor(c_batch).transpose(1, 2).contiguous()
    else:
        c_batch = None

    if global_conditioning:
        g_batch = torch.LongTensor([x[2] for x in batch])
    else:
        g_batch = None

    path_batch = list(x[3] for x in batch)

    video_batch = torch.cat(video_block, 0)
    flow_batch = torch.cat(flow_block, 0)

    # Covnert to channel first i.e., (B, C, T)
    x_batch = torch.FloatTensor(x_batch).transpose(1, 2).contiguous()
    # Add extra axis
    if is_mulaw_quantize(hparams.input_type):
        y_batch = torch.LongTensor(y_batch).unsqueeze(-1).contiguous()
    else:
        y_batch = torch.FloatTensor(y_batch).unsqueeze(-1).contiguous()

    input_lengths = torch.LongTensor(input_lengths)

    return video_batch, flow_batch, c_batch, x_batch, y_batch, g_batch, input_lengths, path_batch



def split_name_for_phase(phase):
    if phase == "train":
        return hparams.train_split_name
    if phase == "val":
        return hparams.val_split_name
    if phase == "test":
        return hparams.test_split_name
    raise ValueError(f"Unknown data phase: {phase}")


def split_has_rows(data_root, split_name):
    split_path = os.path.join(data_root, split_name)
    if not os.path.exists(split_path):
        return False
    with open(split_path, "r", encoding="utf-8") as handle:
        return any(line.strip() for line in handle)


def get_data_loaders(data_root, speaker_id=None, test_shuffle=True, phases=("train", "val")):
    data_loaders = {}
    local_conditioning = hparams.cin_channels > 0
    for phase in phases:
        train = phase == "train"
        split_name = split_name_for_phase(phase)
        if not split_has_rows(data_root, split_name):
            if train:
                raise RuntimeError(f"Training split is missing or empty: {os.path.join(data_root, split_name)}")
            print(f"[{phase}]: split missing or empty, skipping {split_name}")
            continue
        X = FileSourceDataset(RawAudioDataSource(data_root, speaker_id=speaker_id,
                                                 train=train,
                                                 test_size=hparams.test_size,
                                                 phase=phase,
                                                 split_name=split_name))
        Image = FileSourceDataset(ImageSpecDataSource(data_root, speaker_id=speaker_id,
                                                      train=train,
                                                      test_size=hparams.test_size,
                                                      phase=phase,
                                                      split_name=split_name))
        if local_conditioning:
            Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id=speaker_id,
                                                      train=train,
                                                      test_size=hparams.test_size,
                                                      phase=phase,
                                                      split_name=split_name))
            assert len(X) == len(Mel)
            print("Local conditioning enabled. Shape of a sample: {}.".format(
                Mel[0].shape))
        else:
            Mel = None
        print("[{}]: length of the dataset is {}".format(phase, len(X)))

        if train:
            lengths = np.array(X.file_data_source.lengths)
            # Prepare sampler
            sampler = PartialyRandomizedSimilarTimeLengthSampler(
                lengths, batch_size=hparams.batch_size)
            shuffle = True
        else:
            sampler = None
            shuffle = test_shuffle

        dataset = PyTorchImageDataset(X, Mel, Image)

        data_loader = data_utils.DataLoader(
            dataset, batch_size=hparams.batch_size,
            num_workers=hparams.num_workers, shuffle=shuffle,
            collate_fn=collate_fn, pin_memory=hparams.pin_memory)

        speaker_ids = {}

        if len(speaker_ids) > 0:
            print("Speaker stats:", speaker_ids)

        data_loaders[phase] = data_loader

    return data_loaders
