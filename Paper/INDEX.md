# Paper Index — ĐATN CDDFuse Variants

Bộ paper tham khảo cho thesis, tổ chức theo Module cải tiến.

## `foundation/` — Kiến trúc gốc & nền tảng

| File | Tác giả | Venue | Vai trò |
|---|---|---|---|
| `Zhao_CDDFuse_CVPR2023.pdf` | Zhao et al. | CVPR 2023 | **Paper gốc** — cải tiến trực tiếp |
| `Zamir_Restormer_CVPR2022.pdf` | Zamir et al. | CVPR 2022 | Restormer backbone (Encoder/Decoder của CDDFuse) |
| `Vaswani_AttentionIsAllYouNeed_NIPS2017.pdf` | Vaswani et al. | NIPS 2017 | Foundation attention |

## `ModuleA/` — Fusion Rule alternatives (thay `f_I + f_V`)

### A.2 Gated
| File | Idea adopt |
|---|---|
| `Dauphin_GLU_ICML2017.pdf` | GLU `g(x) ⊙ x` — base của Gated formulation |
| `Srivastava_HighwayNetworks_NIPS2015.pdf` | Highway gating `g·H(x) + (1-g)·x` — direct inspiration |

### A.3 CrossAttn
| File | Idea adopt |
|---|---|
| `../foundation/Zamir_Restormer_CVPR2022.pdf` | MDTA channel attention — tái dùng cho cross-modality |
| `Ma_SwinFusion_2022.pdf` | Cross-attention giữa 2 modality (IR/VIS fusion) |

### A.4 ChannelMoE
| File | Idea adopt |
|---|---|
| `Shazeer_MoE_ICLR2017.pdf` | MoE routing foundation |
| `Riquelme_VMoE_NeurIPS2021.pdf` | Vision MoE — sparse expert routing trong CV |
| `Yang_TCMoA_CVPR2024.pdf` | TC-MoA cho task-customized fusion — **direct inspiration** A.4 |

### Liên quan (fusion baselines)
| File | Idea adopt |
|---|---|
| `Li_DenseFuse_TIP2019.pdf` | Saliency-based weighted addition — gần spirit của Gated/CrossAttn |
| `Xu_U2Fusion_PAMI2020.pdf` | Adaptive weighting network — generalize fusion rules |

## `ModuleB/` — Loss Target alternatives (thay `max(I_y, I_ir)`)

> Module B uses concepts mostly từ ModuleA papers (DenseFuse, U2Fusion). Một số reference riêng:

| Concept | Reference |
|---|---|
| B.2 Mean fusion | Cổ điển — non-DL pixel averaging (Toet 1989) |
| B.3 Saliency-weighted | Itti & Koch *PAMI 1998* (gốc); DenseFuse + U2Fusion (apply) |
| B.4 Learnable | U2Fusion adaptive weight network |

**Note**: Itti & Koch 1998 không có trên arXiv (paywalled IEEE PAMI). Nếu cần fulltext, search Google Scholar hoặc Sci-Hub.

## `ModuleC/` — Decomp Loss reformulation

| File | Idea adopt |
|---|---|
| `Kornblith_CKA_ICML2019.pdf` | **CKA** — đã test (C.1), không cải thiện |
| `Bansal_Orthogonality_NeurIPS2018.pdf` | Orthogonality regularization (C.10, chưa thử) |
| `Jiang_FocalFrequencyLoss_ICCV2021.pdf` | Frequency-domain loss (C.11, chưa thử) |

## Older papers ở root (có sẵn từ trước)

| File | Notes |
|---|---|
| `1-s2.0-S0010482523004249-main_deeplearning.pdf` | DL fusion review |
| `1-s2.0-S016516842100075X-main.pdf` | Survey |
| `1-s2.0-S092523121630649X-main_survey_2016.pdf` | Older survey 2016 |
| `liu2012_chisodanhgia.pdf` | Liu 2012 metrics paper |
| `s11042-020-08834-5_2021.pdf` | 2021 paper |

---

## Cách cite trong thesis

Mỗi paper khi cite trong LaTeX dùng key từ `report_latex/bibliography.bib`. Nếu thiếu entry, thêm vào `bibliography.bib` rồi dùng `\cite{key}`.

Ví dụ:
```latex
Mô-đun A.2 sử dụng gated mechanism (GLU) \cite{dauphin2017glu} kết hợp Highway 
concept \cite{srivastava2015highway} áp dụng cho fusion layer của CDDFuse 
\cite{zhao2023cddfuse}.
```

## Quick reference — Concept → Paper mapping

| Bạn cần cite | Paper |
|---|---|
| CDDFuse architecture | Zhao 2023 |
| Restormer attention | Zamir 2022 |
| Gated mechanism | Dauphin 2017 |
| Highway / soft gating | Srivastava 2015 |
| Transformer / attention | Vaswani 2017 |
| MoE | Shazeer 2017 (foundation), Riquelme 2021 (vision), Yang 2024 (fusion) |
| Saliency-weighted fusion | Li DenseFuse 2019 |
| Adaptive fusion weights | Xu U2Fusion 2020 |
| Cross-modality fusion attn | Ma SwinFusion 2022 |
| CKA similarity | Kornblith 2019 |
| Orthogonality reg | Bansal 2018 |
| Frequency loss | Jiang 2021 |
