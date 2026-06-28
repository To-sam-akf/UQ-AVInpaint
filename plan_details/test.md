**1. 准备 demo split**
```bash
cd /root/EC-ViAv-vgpu
export DATA_ROOT=/root/shared-nvme/data

python - <<'PY'
from pathlib import Path
import re

split = Path("/root/shared-nvme/data/test_av_split.txt")
out = Path("/root/shared-nvme/data/test_demo_5_instruments_split.txt")

wanted = ["accordion", "acoustic_guitar", "flute", "saxophone", "xylophone"]
selected = {}
fallback = {}

for line in split.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    m = re.search(r"processed/([^/]+)/", line)
    if not m:
        continue
    inst = m.group(1)
    fallback.setdefault(inst, line)
    if inst in wanted and inst not in selected:
        selected[inst] = line

rows = []
for inst in wanted:
    if inst in selected:
        rows.append(selected[inst])

if len(rows) < 5:
    for inst, line in fallback.items():
        if inst not in selected:
            rows.append(line)
        if len(rows) >= 5:
            break

out.write_text("\n".join(rows) + "\n")
print(f"wrote {out}")
for row in rows:
    print(row)
PY
```
确认是 5 种乐器：
```
cat "$DATA_ROOT/test_demo_5_instruments_split.txt" | cut -d/ -f2
```

**2. 设置 checkpoint 和 semantic evidence**
```bash
export SEM_DIR="$DATA_ROOT/semantic_evidence/clip_vit_b32"
export TEST_SEM="$SEM_DIR/test_av_split.jsonl"

export FINAL_CKPT=checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected/EC-VIAI-AV-PatchGAN_checkpoint_step000100000.pth.tar
export DEMO_ROOT=checkpoints/demo_k_candidates_final_5instruments
```

**3. 导出 normal video 下的 K 个候选 Mel 和 wav**
```bash
python test_viai_av.py \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --enable_candidate_scorer \
  --evidence_source fused \
  --semantic_evidence_path "$TEST_SEM" \
  --semantic_evidence_weight 0.35 \
  --semantic_missing_score 0.0 \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.2 \
  --lambda_calib 0.05 \
  --calib_error_tau 0.1 \
  --resume_path "$FINAL_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_demo_5_instruments_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --results_dir "$DEMO_ROOT/none" \
  --video_perturbation none \
  --save_candidates \
  --use_vocoder \
  --vocoder_max_samples 5 \
  --vocoder_n_iter 128 \
  --calibration_bins 10 \
  --display_id 0
```

**4. 输出位置**

Top-1 Mel 图：

```bash
ls "$DEMO_ROOT/none/mel-image/step000100000"
```

K 个候选 Mel 图：

```bash
find "$DEMO_ROOT/none/mel-candidates/step000100000" -type f | head -n 30
```

目录结构大概是：

```text
mel-candidates/step000100000/candidate_00/
mel-candidates/step000100000/candidate_01/
mel-candidates/step000100000/candidate_02/
mel-candidates/step000100000/candidate_03/
```

Top-1 wav：

```bash
find "$DEMO_ROOT/none/wav/step000100000" -type f | head
```

K 个候选 wav：

```bash
find "$DEMO_ROOT/none/wav-candidates/step000100000" -type f | head -n 40
```

每个候选的音频在：

```text
wav-candidates/step000100000/candidate_00/
wav-candidates/step000100000/candidate_01/
wav-candidates/step000100000/candidate_02/
wav-candidates/step000100000/candidate_03/
```

注意：这里 wav 是 Griffin-Lim 从 Mel 反推的，适合展示“候选差异”和 demo，不等价于高质量神经 vocoder 音质。