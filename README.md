<div align="center">

# Homophobia and Transphobia Span Identification<br>in Low-Resource Languages

<p>
  <a href="https://doi.org/10.1016/j.nlp.2025.100169">
    <img src="https://img.shields.io/badge/DOI-10.1016%2Fj.nlp.2025.100169-blue?style=for-the-badge&logo=doi" alt="DOI">
  </a>
  <a href="https://www.sciencedirect.com/science/article/pii/S2949719125000457">
    <img src="https://img.shields.io/badge/Journal-NLP%20Journal%202025-4a90d9?style=for-the-badge&logo=elsevier" alt="Journal">
  </a>
  <img src="https://img.shields.io/badge/Languages-Tamil%20%7C%20English%20%7C%20Marathi-orange?style=for-the-badge" alt="Languages">
  <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=for-the-badge" alt="License">
</p>

<!-- <p>
  <a href="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python">
    <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python" alt="Python">
  </a>
  <a href="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/🤗%20HuggingFace-Transformers-yellow?style=flat-square" alt="HuggingFace">
  </a>
</p> -->

**Prasanna Kumar Kumaresan¹ · Devendra Deepak Kayande² · Ruba Priyadharshini³ · Paul Buitelaar¹ · Bharathi Raja Chakravarthi¹**

¹ Data Science Institute, University of Galway, Ireland &nbsp;|&nbsp; ² Indian Institute of Information Technology, Allahabad, Uttar Pradesh, India &nbsp;|&nbsp; ³ Gandhigram Rural Institute-Deemed to be University, Tamil Nadu, India

*(Natural Language Processing Journal, Volume 12, 2025, Article 100169)*

</div>

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Dataset](#-dataset)
  - [Dataset Creation Pipeline](#dataset-creation-pipeline)
  - [Label Distribution in Source Data](#label-distribution-in-source-data)
- [Methodology](#-methodology)
  - [Model Architecture](#model-architecture)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---
> 🔬 *Disclaimer: This work is for educational and research purposes only and is intended to classify and understand hate speech in a structured research framework. The examples do not represent the views of the authors or their affiliated institutions.*
---
## 📌 Abstract

Online platforms have become prevalent because they promote free speech and group discussions. However, they also serve as platforms for hate speech, which can negatively impact the psychological well-being of vulnerable people. This is especially true for members of the LGBTQ+ community, who are often the targets of homophobia and transphobia in online environments.

Our study makes **three main contributions**:

| # | Contribution |
|---|---|
| 1️⃣ | **Novel multilingual span dataset** with span-level annotations for homophobia and transphobia in **Tamil, English, and Marathi**, collected from real-world YouTube comments |
| 2️⃣ | **BERT-based models** with CRF and BiLSTM layers for fine-grained, token-level span detection of harmful content |
| 3️⃣ | **Comprehensive benchmarking** of monolingual vs. multilingual models for detecting subtle forms of hate speech across low-resource languages |

---
## 📂 Repository Structure

```text
HT-Span-2025/
│
├── HT_Span_GOLD dataset/                # Gold-standard annotated datasets
│   ├── HT_Span_English_GOLD.csv         # English gold dataset
│   ├── HT_Span_Marathi_GOLD.csv         # Marathi gold dataset
│   └── HT_Span_Tamil_GOLD.csv           # Tamil gold dataset
│
├── span-main/                            # Main codebase
│   │
│   ├── data/                             # Data preparation and preprocessing
│   │   ├── add_spans.py
│   │   ├── data_prep.py
│   │   ├── english_span_updated.csv
│   │   ├── final_tamil.csv
│   │   └── marathi_combined_updated_data.csv
│   │
│   └── src/                              # Model training and evaluation
│       │
│       ├── models/                       # BERT-based model architectures
│       │   ├── bert_backbone.py
│       │   ├── bert_bilstm_crf.py
│       │   └── bert_crf.py
│       │
│       ├── span_dataset.py               # Dataset handling
│       ├── train_and_evaluate.py         # Training and evaluation pipeline
│       └── utils.py                      # Utility functions
│
├── .gitignore
└── README.md
```

## 🗂️ Dataset

### Dataset Creation Pipeline

The dataset was built on top of prior research ([Chakravarthi, 2024](https://doi.org/10.1007/s41060-023-00432-8); [Kumaresan et al., 2023](https://doi.org/10.1016/j.nlp.2023.100041)), adding **span-level annotations** on top of existing sentence-level labels.

Data was collected from YouTube comments on LGBTQ+-related videos using the [YouTube Comment Scraper](https://github.com/ahmedshahriar/youtube-comment-scraper) tool.

**Pipeline steps:**
1. Source comments from 3-class, 5-class, and 7-class labelled datasets
2. Filter only **Homophobia (H)** and **Transphobia (T)**  from 3-class - labelled comments
3. Export filtered comments to text line files
4. Annotate spans via Doccano → export as JSON
5. Convert to CSV → **HT_Span_GOLD dataset**

---

#### Annotation Labels

| Label | Definition | Examples |
|---|---|---|
| **Homophobia** | Hostile language targeting LGBTQ+ individuals based on sexual orientation | Terms like *homo*, *fag*, *dyke*; phrases like *"gays deserve to be shot dead"* |
| **Transphobia** | Demeaning or violent language targeting transgender individuals | Terms like *she-male*, *tranny*; phrases like *"kill those trannies"* |
| **None of the above** | Neutral or unrelated content | Any text not targeting LGBTQ+ identities |

---

## 🤖 Methodology

### Model Architecture

We implemented three BERT-based architectures for multilingual span detection:

> *The architecture combines a fine-tuned BERT backbone (pre-trained on hate speech + NER data) with either: a span detection head (Model 1), a CRF layer (Model 2), or a BiLSTM + CRF combination (Model 3).*

| Architecture | Key Idea |
|---|---|
| **BERT + Span Head** | Token-wise sigmoid outputs for span detection; [CLS] pooling for classification |
| **BERT + CRF** | CRF models tag transition probabilities — better sequence coherence |
| **BERT + BiLSTM + CRF** | BiLSTM captures forward + backward context before CRF decoding |

---

## 📖 Citation

If you use this dataset, or findings in your research, please cite:

```bibtex
@article{KUMARESAN2025100169,
  title    = {Homophobia and transphobia span identification in low-resource languages},
  journal  = {Natural Language Processing Journal},
  volume   = {12},
  pages    = {100169},
  year     = {2025},
  issn     = {2949-7191},
  doi      = {https://doi.org/10.1016/j.nlp.2025.100169},
  url      = {https://www.sciencedirect.com/science/article/pii/S2949719125000457},
  author   = {Prasanna Kumar Kumaresan and Devendra Deepak Kayande and
              Ruba Priyadharshini and Paul Buitelaar and Bharathi Raja Chakravarthi},
  keywords = {LGBTQ+ hate speech, Span-based classification, Low-resource language,
              BERT architecture, Sequence labeling},
  abstract = {Online platforms have become prevalent because they promote free speech
              and group discussions. However, they also serve as platforms for hate speech,
              which can negatively impact the psychological well-being of vulnerable people.
              This is especially true for members of the LGBTQ+ community, who are often
              the targets of homophobia and transphobia in online environments. Our study
              makes three main contributions: (1) we developed a new dataset with span-level
              annotations for homophobia and transphobia in Tamil, English, and Marathi;
              (2) we employed advanced language models using BERT-based architectures,
              Conditional Random Field (CRF), and Bidirectional Long Short-Term Memory
              (BiLSTM) layers to enhance span-level detection of harmful content; and
              (3) we conducted benchmarking to evaluate the effectiveness of monolingual
              and multilingual models in detecting subtle forms of hate speech.}
}
```

---

## 🙏 Acknowledgements

- **Bharathi Raja Chakravarthi** was funded by Research Ireland under grant **SFI/12/RC/2289_P2** (Insight) and supported by **Google Gemma Awards**.
- This work was supported by the **Centre for Research Training in Artificial Intelligence** (grant **SFI/18/CRT/6223**) and the College of Science and Engineering, University of Galway, Ireland.

---

<div align="center">

Made with ❤️ for safer, more inclusive online spaces

**Kumaresan et al. 2025**

</div>
