# SceneScript — AI That Watches Videos and Tells You What Happens

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aayushx9/SceneScript-video-understanding/blob/main/SceneScript%20-%20Video%20Understanding.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![CLIP](https://img.shields.io/badge/Vision-CLIP%20ViT--B%2F32-orange)](https://github.com/openai/CLIP)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://github.com/Aayushx9/SceneScript-video-understanding/blob/main/LICENSE)

**Author:** Aayush Bharadwaj · [github.com/Aayushx9](https://github.com/Aayushx9)
**MS Data Science · University of Colorado Boulder**

> Multimodal video understanding pipeline. CLIP ViT-B/32 reads every sampled frame, detects scene cuts by embedding similarity, and FLAN-T5-Large stitches the observations into timestamped, plain-English descriptions — running end-to-end in under a minute on a free Colab T4 GPU. Ships as a Gradio dashboard: upload any video, get a structured breakdown.

---

## What It Does

SceneScript watches a video the way a person skimming security footage would — sampling frames, recognizing what's in each one, noticing when the scene changes, and writing up what happened. No fine-tuning, no labeled training set: CLIP's zero-shot classification does the seeing, and FLAN-T5 does the writing.

**One-liner:**
*"I built a multimodal pipeline that extracts frames from any video, runs CLIP zero-shot classification across an 75-label vocabulary to understand each frame, detects scene cuts from embedding cosine similarity, and chains a FLAN-T5 language model to generate timestamped natural-language summaries — deployed as a Gradio app that processes video faster than real time."*

---

## Pipeline Architecture

```
Video file
    │
    ▼
Frame Extraction (OpenCV)         — 1 frame every N seconds, resized to 336×336
    │
    ▼
CLIP ViT-B/32                     — encodes each frame; zero-shot scores against
    │                                75 labels (51 scenes · 14 objects · 10 moods)
    ▼
Scene Change Detection            — cosine similarity between consecutive
    │                                frame embeddings, thresholded
    ▼
Segment Grouping                  — frames between scene cuts become one segment
    │
    ▼
FLAN-T5-Large                     — writes a 2–3 sentence description per segment
    │                                from its CLIP-observed labels
    ▼
Timestamped Summary + Overall Paragraph
    │
    ▼
Gradio Dashboard                  — confidence timeline, scene map, label
                                     frequency, PCA embedding view
```

| Section | Step                        | Description                                                      |
| ------- | --------------------------- | ------------------------------------------------------------------ |
| 1–2     | Setup                       | Install dependencies, load CLIP ViT-B/32 + FLAN-T5-Large            |
| 3       | Frame extraction            | Sample frames at a fixed time interval                             |
| 4       | CLIP scene understanding    | Zero-shot classification + frame embeddings                        |
| 5       | Scene change detection      | Cosine similarity between consecutive embeddings                   |
| 6       | Language model summaries    | FLAN-T5 turns CLIP labels into natural-language segment descriptions |
| 7       | Full pipeline               | End-to-end `analyse_video()` function                              |
| 8       | Visualizations               | Confidence timeline, scene Gantt chart, label frequency, PCA plot  |
| 9       | Gradio demo                 | Upload-a-video dashboard                                           |
| 10      | Custom label scoring         | Score any frame against a domain-specific label set                |

**Runtime:** ~20s to fully analyze a 30s clip on a Colab T4 GPU (**1.5× faster than real time**).

---

## Models

| Model            | Role                                            | Size    |
| ----------------- | ------------------------------------------------ | ------- |
| CLIP ViT-B/32     | Vision encoder — understands each frame           | ~340 MB |
| FLAN-T5-Large     | Language model — writes the natural-language summary | ~780 MB |

Both run free on a Colab T4 GPU — no API keys required.

---

## Label Vocabulary

75 zero-shot labels spanning three categories, so CLIP can describe a video's setting, actions, and content without ever seeing training examples for it:

```
Scenes  (51): indoor/outdoor, urban street, forest, beach, office, kitchen,
              stadium, classroom, hospital, parking lot, ...
Objects (14): text/title card, product close-up, map or diagram, ...
Moods   (10): (tone/atmosphere labels layered on top of scene + object)
```

---

## Sample Run — 30s Test Clip

```
Video info       : 30.0 FPS · 911 frames · 30.4s duration
Sampling         : 1 frame every 2.5s → 13 frames extracted
CLIP analysis    : 0.8s for 13 frames
Scene segments   : 1 detected (single continuous scene)
Total pipeline   : 20.0s
Realtime factor  : 1.5x faster than the video itself

Segment 1 [00:00 → 00:30]:
  "A person walks through a wooded area."
  Tags: outdoor scene · person walking · forest or nature
  Avg CLIP confidence: 29.9%
```

---

## Custom Label Scoring — Domain Adaptation Demo

Beyond the built-in 75-label vocabulary, any frame can be scored against a custom label set on the fly — no retraining. Tested against a sports-domain vocabulary on a single frame:

```
half-time break                      27.1%  ████████████████████████████
replay or slow motion                25.5%  ███████████████████████████
player running with ball             19.5%  █████████████████████
goal or score being made             13.0%  ██████████████
referee making a call                 8.5%  █████████
trophy or celebration                 3.4%  ████
```

This is the same mechanism that would adapt the pipeline to medical footage, retail security, or any other specialized domain — swap the label list, get zero-shot scores immediately.

---

## Interactive Demo — Gradio Dashboard

Upload any video and get back:

- A timestamped, plain-English scene breakdown
- Scene segmentation with per-segment confidence
- A CLIP confidence timeline with scene-cut markers
- Label frequency analytics across the whole video
- A PCA plot of CLIP frame embeddings, colored by detected scene

---

## Key Results

| Metric                          | Value                          |
| -------------------------------- | -------------------------------- |
| Vision encoder                   | CLIP ViT-B/32 (zero-shot)        |
| Language model                   | FLAN-T5-Large                    |
| Label vocabulary                 | 75 labels (51 scene / 14 object / 10 mood) |
| Test clip                        | 30s, 911 frames @ 30fps          |
| Frames sampled                   | 13 (1 every 2.5s)                |
| CLIP inference time              | 0.8s for 13 frames               |
| End-to-end pipeline time         | 20.0s                            |
| Realtime factor                  | 1.5× faster than the source video |
| GPU                               | Colab T4 (15.6 GB VRAM)          |

---

## Quick Start

### Option 1 — Run in Colab (Recommended)

Click the **Open in Colab** badge at the top. Set the runtime to **T4 GPU** (`Runtime → Change runtime type → T4 GPU`) and run all cells top to bottom.

### Option 2 — Run Locally

```
git clone https://github.com/Aayushx9/SceneScript-video-understanding.git
cd SceneScript-video-understanding
pip install -r requirements.txt
jupyter notebook "SceneScript - Video Understanding.ipynb"
```

A CUDA-capable GPU is strongly recommended — FLAN-T5-Large generation is slow on CPU.

---

## Tech Stack

| Category         | Tools                                           |
| ------------------ | -------------------------------------------------|
| Vision             | OpenAI CLIP (ViT-B/32)                          |
| Language Model     | FLAN-T5-Large (Hugging Face Transformers)       |
| Video Processing   | OpenCV, ffmpeg                                  |
| Dashboard          | Gradio, Plotly                                  |
| Dimensionality Reduction | Scikit-learn (PCA)                        |
| Data Processing    | Pandas, NumPy                                   |

---

## Target Companies

Directly relevant to video and content-understanding teams at:
**YouTube** · **Netflix** · **Meta** · **Google** · **Apple** · **Tesla** · **NVIDIA**

(YouTube alone receives 500 hours of video uploaded per minute; Netflix's catalog runs 15,000+ titles needing structured metadata — both are exactly the scale problem this pipeline is built to address.)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Aayush Bharadwaj](https://github.com/Aayushx9) · MS Data Science, University of Colorado Boulder*
