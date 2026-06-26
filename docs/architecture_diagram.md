# EC-VIAI-AV 架构文档

## 整体 Pipeline 概览

下图展示从输入到输出的完整数据流主线。

```mermaid
flowchart LR
    subgraph Input["输入"]
        AUDIO["Audio Mel-spectrogram"]
        VIDEO["Video + Optical Flow"]
    end

    subgraph Encoding["编码"]
        ME["Mel Encoder"]
        VE["VideoEncoder"]
    end

    subgraph Evidence["证据估计"]
        HEE["VisualEvidenceEstimator<br/>heuristic evidence"]
        SE["CLIP Semantic Table<br/>semantic evidence"]
        FUSE["证据融合 evidence_score<br/>fused = heuristic + weight"]
    end

    subgraph Gate["门控融合"]
        EG["EvidenceFusionGate<br/>gate = sigmoid MLP(audio, video, evidence)<br/>calibrated = gate x video + (1-gate) x prior"]
    end

    subgraph Sample["多候选采样"]
        BA["BottleneckAdapter Stochastic<br/>mu, logvar to sample K=4 latent"]
    end

    subgraph Decode["解码 K 候选"]
        DEC["Mel Decoder<br/>K 个候选并行解码<br/>to K Mel reconstructions"]
    end

    subgraph Select["排序选择和不确定性"]
        CS["CandidateScorer<br/>pi = softmax based on<br/>proxy L1 + boundary + sync"]
        UH["UncertaintyHead<br/>input: audio, video, 7-dim stats<br/>to uncertainty 0-1"]
        TOP1["gather top-1 from K Mel"]
    end

    subgraph Output["输出"]
        RECON["最终 Mel-spectrogram"]
    end

    AUDIO --> ME
    VIDEO --> VE
    VIDEO --> HEE
    VE --> HEE

    ME --> |mel_bottleneck| EG
    VE --> |video_feature| EG

    HEE --> FUSE
    SE --> FUSE
    FUSE --> |evidence_score| EG
    FUSE --> |evidence_score| CS
    FUSE --> |evidence_score| UH

    EG --> |calibrated_video| BA
    ME --> BA

    BA --> |K latent residuals<br/>added to mel_features| DEC

    DEC --> |K candidate Mel| CS
    DEC --> |K candidate Mel| TOP1

    CS --> |top-1 index| TOP1
    CS --> |pi distribution| Select

    UH --> |uncertainty| Output

    TOP1 --> RECON

    style Input fill:#ffebee,stroke:#b71c1c,color:#000
    style Encoding fill:#e1f5fe,stroke:#0288d1,color:#000
    style Evidence fill:#fff3e0,stroke:#f57c00,color:#000
    style Gate fill:#f3e5f5,stroke:#7b1fa2,color:#000
    style Sample fill:#e8f5e9,stroke:#388e3c,color:#000
    style Decode fill:#f9fbe7,stroke:#afb42b,color:#000
    style Select fill:#fce4ec,stroke:#c2185b,color:#000
    style Output fill:#e0f7fa,stroke:#00695c,color:#000
```

**关键数据流（对应代码 `_forward_inpainter` line 840-950 + `_score_and_select_candidates` line 757-838）：**

1. **音频** → Mel Encoder → mel_bottleneck (256ch)
2. **视频 + 光流** → VideoEncoder → video_feature (256ch)
3. **video_feature** → VisualEvidenceEstimator → heuristic_evidence (0~1)
4. **heuristic + semantic** → 证据融合 → evidence_score
5. **mel_bottleneck + video_feature + evidence_score** → EvidenceFusionGate
   - gate = sigmoid(MLP(audio, video, evidence_score)) → gate_value (0~1)
   - calibrated_video = gate × video + (1-gate) × audio_prior
6. **mel_bottleneck + calibrated_video** → BottleneckAdapter.sample_residuals()
   - 采样 K=4 个 latent residual → 加到 mel_features[-1] → K 个不同的 bottleneck feature
7. **K 个 bottleneck feature** → Mel_Decoder (并行 B*K) → **K 个候选 Mel 图**
8. **K 个候选 Mel + evidence_score** → CandidateScorer (基于 proxy L1 + boundary + sync 三个统计量) → π 分布
9. **K 个候选 Mel + top-1 index** → gather → **最终重建 Mel**
10. **同时 UncertaintyHead(audio, video, 7维统计量)** → uncertainty 分数 (0~1)

---

## 详细架构图

```mermaid
flowchart TB
    subgraph Inputs["输入数据"]
        V["Video Frames image_crop"]
        F["Optical Flow"]
        A["Audio Mel-spectrogram"]
    end

    subgraph Encoders["编码器"]
        VE["VideoEncoder ImageEmbedding"]
        ME["Mel Encoder MelEncoder"]
    end

    subgraph Evidence["证据估计 Evidence"]
        direction TB
        HEE["VisualEvidenceEstimator<br/>启发式证据"]
        subgraph HEE_Internals["启发式证据计算"]
            FM["Flow Magnitude<br/>光流幅度"]
            TS["Temporal Signal<br/>时间变化信号"]
            SC["Sync Score<br/>音画同步分数"]
            FS["Feature Score<br/>特征强度"]
        end
        SE["SemanticEvidenceTable<br/>语义证据表 CLIP"]
        EF["证据融合 evidence_score<br/>fused = heuristic + weight"]
    end

    subgraph GateModule["门控模块 EvidenceFusionGate"]
        AP["Audio Prior<br/>音视频先验"]
        GM["Gate MLP sigmoid"]
        CF["校准视频特征<br/>gate x video + (1-gate) x prior"]
    end

    subgraph Adapter["随机适配器 BottleneckAdapter"]
        STO["Stochastic Adapter<br/>mu, logvar to sample K=4 latent"]
        SS["Sigma Scaling<br/>evidence 缩放 sigma"]
    end

    subgraph DecoderK["解码器 MelDecoderImage"]
        MD["K 个候选并行解码 B*K"]
    end

    subgraph ScorerUncertainty["排序器和不确定性"]
        CS["CandidateScorer<br/>pi = softmax based on<br/>proxy L1 + boundary + sync"]
        UH["UncertaintyHead<br/>input: audio, video, 7-dim stats<br/>sigmoid to uncertainty"]
    end

    subgraph Gather["输出选择"]
        GT["gather top-1 from K Mel"]
    end

    subgraph Losses["损失函数"]
        L1["L1 Reconstruction Loss"]
        GAN["GAN Loss MelDiscriminator"]
        MK["min_k Loss top-K 候选最小"]
        MK2["mean_k Loss"]
        BD["boundary Loss"]
        DV["diversity Loss"]
        GE["gate_evidence Loss SmoothL1"]
        CSL["candidate_scorer Loss"]
        UC["uncertainty_calib Loss SmoothL1"]
        SYNC["Sync Contrastive Loss"]
    end

    subgraph Outputs["输出"]
        REC["重建 Mel-spectrogram"]
        GATE["gate_value 0 to 1"]
        UNC["uncertainty 0 to 1"]
        PI["candidate pi 排序概率"]
    end

    V --> VE
    V --> F
    A --> ME
    F --> HEE
    VE --> HEE

    ME --> |mel_bottleneck| GateModule
    VE --> |video_feature| GateModule

    HEE --> |heuristic| EF
    SE --> |semantic| EF
    EF --> |evidence_score| GateModule
    EF --> |evidence_score| ScorerUncertainty

    GateModule --> |calibrated_video| Adapter
    GateModule --> GATE

    ME --> Adapter
    Adapter --> |K latent residuals| DecoderK
    ME --> DecoderK

    DecoderK --> |K candidate Mel| ScorerUncertainty
    DecoderK --> |K candidate Mel| Gather

    ScorerUncertainty --> |top-1 index| Gather
    ScorerUncertainty --> PI
    ScorerUncertainty --> UNC

    Gather --> REC

    REC --> L1
    REC --> MK
    REC --> MK2
    REC --> BD
    REC --> GAN
    REC --> DV
    GateModule --> GE
    ScorerUncertainty --> CSL
    ScorerUncertainty --> UC
    VE --> SYNC
    ME --> SYNC

    classDef encoder fill:#e1f5fe,stroke:#0288d1
    classDef evidence fill:#fff3e0,stroke:#f57c00
    classDef gate fill:#f3e5f5,stroke:#7b1fa2
    classDef adapter fill:#e8f5e9,stroke:#388e3c
    classDef decoder fill:#f9fbe7,stroke:#afb42b
    classDef scorer fill:#fce4ec,stroke:#c2185b
    classDef gather fill:#ffccbc,stroke:#bf360c
    classDef loss fill:#efebe9,stroke:#5d4037
    classDef output fill:#e0f7fa,stroke:#00695c
    classDef input fill:#ffebee,stroke:#b71c1c

    class VE,ME encoder
    class HEE,SE,EF evidence
    class GateModule,AP,GM,CF gate
    class Adapter,STO,SS adapter
    class MD decoder
    class CS,UH scorer
    class GT gather
    class L1,GAN,MK,MK2,BD,DV,GE,CSL,UC,SYNC loss
    class REC,GATE,UNC,PI output
    class V,F,A input
```

---

## 推理流程 (Test-time)

```mermaid
flowchart TD
    START(["input audio + video + flow"]) --> VE[Video Encoder]
    START --> ME[Mel Encoder]
    START --> HEE["VisualEvidenceEstimator<br/>heuristic_evidence"]

    HEE --> EF{"evidence_source?"}

    EF -->|heuristic| EF_H["evidence = heuristic"]
    EF -->|semantic| EF_S["evidence = semantic<br/>lookup CLIP JSONL"]
    EF -->|fused| EF_F["evidence = heuristic + weight"]

    ME --> |mel_bottleneck| FUSION[EvidenceFusionGate]
    VE --> |video_feature| FUSION
    EF --> FUSION

    FUSION --> |calibrated_video| ADAPTER[BottleneckAdapter]
    ME --> |mel_bottleneck| ADAPTER

    ADAPTER --> |K latent residuals| DECODER["Mel Decoder<br/>B*K parallel decode"]
    ME --> DECODER

    DECODER --> |K candidate Mel| SCORER["CandidateScorer<br/>pi distribution"]
    DECODER --> |K candidate Mel| SELECT["gather top-1"]

    SCORER --> |top-1 index| SELECT
    EF --> |evidence_score| SCORER

    SCORER --> UNCERTAINTY["UncertaintyHead<br/>uncertainty score"]

    SELECT --> OUTPUT(["final reconstructed Mel"])

    UNCERTAINTY --> OUTPUT

    style START fill:#e8f5e9,stroke:#2e7d32
    style OUTPUT fill:#e1f5fe,stroke:#0277bd
```

---

## 训练流水线 (Stage 6→10)

```mermaid
flowchart LR
    subgraph Stage6["Stage6 Multi-Candidate"]
        S6["Stochastic Adapter<br/>K=4 candidates<br/>min_k mean_k boundary loss"]
    end

    subgraph Stage7["Stage7 Evidence Gate + Sigma"]
        S7["开启 EvidenceFusionGate<br/>证据缩放 sigma<br/>gate_evidence loss"]
    end

    subgraph Stage8["Stage8 Scorer + Uncertainty"]
        S8A["Stage8-A Scorer/Uncertainty head<br/>5k steps"]
        S8B["Stage8-B full finetune<br/>to 90k steps"]
    end

    subgraph Stage10["Stage10 Semantic Evidence"]
        S10A["离线 CLIP precompute"]
        S10B["Stage10-B readonly<br/>heuristic/semantic/fused 对比"]
        S10C["Stage10-C Fused train<br/>to 100k steps"]
    end

    Stage6 -->|60000 steps| Stage7
    Stage7 -->|80000 steps| Stage8A
    Stage8A -->|65000 steps| Stage8B
    Stage8B -->|90000 steps| Stage10
    S10A --> S10B
    S10B --> S10C

    classDef s6 fill:#e1f5fe,stroke:#0288d1
    classDef s7 fill:#f3e5f5,stroke:#7b1fa2
    classDef s8 fill:#e8f5e9,stroke:#388e3c
    classDef s10 fill:#fff3e0,stroke:#f57c00

    class S6 s6
    class S7 s7
    class S8A,S8B s8
    class S10A,S10B,S10C s10
```

---

## 模块参数说明

| 模块 | 类名 | 输入 | 输出 | 参数 |
|------|------|------|------|------|
| 视频编码器 | `ImageEmbedding` | 视频帧 | video_feature (256ch) | - |
| 梅尔编码器 | `MelEncoder` | Mel-spectrogram | mel_bottleneck (256ch) | - |
| 启发式证据 | `VisualEvidenceEstimator` | flow, video_feature, audio_feature | heuristic_evidence (0~1) | 无参数（规则驱动） |
| 语义证据表 | `SemanticEvidenceTable` | sample_dir | semantic_evidence | weight=0.35, missing_score=0.0 |
| 证据融合门 | `EvidenceFusionGate` | audio_bottleneck, video_feature, evidence_score | calibrated_video, gate_value | 256ch hidden |
| 瓶颈适配器 | `BottleneckAdapter` | mel_bottleneck, video_feature | K latent residuals | 256ch, scale |
| 梅尔解码器 | `MelDecoderImage` | mel_features + K latent residuals | K candidate Mel | - |
| 候选排序器 | `CandidateScorer` | candidate_stats, audio, video, evidence | logits, pi | 256ch hidden |
| 不确定性头 | `UncertaintyHead` | audio, video, 7-dim stats | uncertainty (0~1) | 256ch hidden |

---

## 关键参数命令行标志

| CLI 标志 | 功能 | 阶段 |
|----------|------|------|
| `--use_gan` | 启用 PatchGAN 判别器 | All |
| `--enable_ec_viai_av` | 启用 EC 模块 | All |
| `--stochastic_adapter` | 随机适配器（K 候选） | Stage6+ |
| `--enable_evidence_gate` | 证据门控 | Stage7+ |
| `--enable_evidence_scaled_sigma` | evidence 缩放 sigma | Stage7+ |
| `--enable_candidate_scorer` | 候选排序器 + 不确定性 | Stage8+ |
| `--evidence_source` | heuristic/semantic/fused | Stage10 |
| `--semantic_evidence_path` | CLIP 预计算 JSONL 路径 | Stage10 |
| `--semantic_evidence_weight` | 语义证据融合权重 (0.35) | Stage10 |