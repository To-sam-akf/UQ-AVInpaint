import json
import os

import torch


VALID_EVIDENCE_SOURCES = ("none", "heuristic", "semantic", "fused")


def strip_trailing_start_index(path):
    path = os.path.normpath(str(path))
    if os.path.basename(path).isdigit():
        return os.path.dirname(path)
    return path


def normalize_sample_dir(path, data_root=None):
    path = strip_trailing_start_index(path)
    if data_root is not None and not os.path.isabs(path):
        path = os.path.join(str(data_root), path)
    return os.path.abspath(os.path.normpath(path))


def relative_sample_dir(path, data_root=None):
    path = normalize_sample_dir(path, data_root=data_root)
    if data_root is None:
        return None
    data_root = os.path.abspath(os.path.normpath(str(data_root)))
    try:
        return os.path.relpath(path, data_root)
    except ValueError:
        return None


def sample_dir_keys(path, data_root=None):
    absolute = normalize_sample_dir(path, data_root=data_root)
    keys = [absolute]
    relative = relative_sample_dir(absolute, data_root=data_root)
    if relative is not None and not relative.startswith(".."):
        keys.append(os.path.normpath(relative))
    return keys


def infer_instrument_from_sample_dir(path):
    parts = os.path.normpath(strip_trailing_start_index(path)).split(os.sep)
    for marker in ("processed", "processed_viai_a"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    raise ValueError(f"Cannot infer instrument from sample path: {path}")


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


class SemanticEvidenceTable:
    def __init__(self, path=None, data_root=None, missing_score=0.0):
        self.data_root = data_root
        self.missing_score = clamp01(missing_score)
        self.records = {}
        self.loaded = False
        if path:
            if data_root is not None and not os.path.isabs(path) and not os.path.exists(path):
                path = os.path.join(str(data_root), path)
            self.path = path
            self.load(path)
        else:
            self.path = None

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Semantic evidence file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "sample_dir" not in record:
                    raise ValueError(
                        f"Missing sample_dir in semantic evidence record {line_number}: {path}"
                    )
                record["semantic_score"] = clamp01(record.get("semantic_score", 0.0))
                for key in sample_dir_keys(record["sample_dir"], data_root=self.data_root):
                    self.records[key] = record
        self.loaded = True
        return self

    def lookup_record(self, path):
        for key in sample_dir_keys(path, data_root=self.data_root):
            record = self.records.get(key)
            if record is not None:
                return record
        return None

    def lookup_score(self, path):
        record = self.lookup_record(path)
        if record is None:
            return self.missing_score
        return clamp01(record.get("semantic_score", self.missing_score))

    def lookup_scores(self, paths):
        return [self.lookup_score(path) for path in paths]


def semantic_scores_to_tensor(scores, reference):
    return torch.tensor(
        scores,
        device=reference.device,
        dtype=reference.dtype,
    ).view(-1, 1)


def combine_evidence_scores(heuristic, semantic, source="none", weight=0.35):
    if source not in VALID_EVIDENCE_SOURCES:
        raise ValueError(
            f"Unsupported evidence source: {source}. "
            f"Expected one of {', '.join(VALID_EVIDENCE_SOURCES)}."
        )
    if source in {"none", "heuristic"}:
        return heuristic
    semantic = semantic.to(device=heuristic.device, dtype=heuristic.dtype)
    if source == "semantic":
        return torch.clamp(semantic, min=0.0, max=1.0)
    weight = clamp01(weight)
    return torch.clamp((1.0 - weight) * heuristic + weight * semantic, min=0.0, max=1.0)
