# LFA-CIFAR10: Loss Function Assessment (LFA) and F‑ECE on CIFAR‑10

## Overview


This repository provides an example of reproducible implementation of the **Loss Function Assessment (LFA)** framework and the **Focal‑balanced Exponential Cross‑Entropy (F‑ECE)** loss on **CIFAR‑10**. 
The goal is to analyze hard‑to‑learn classes and to evaluate how probability‑defined losses (BCE, Focal Loss, ASL, F‑ECE) shape gradient–probability behavior and recall on difficult categories.

## Repository Structure

```


LFA-CIFAR10/
├── datasets/          # CIFAR-10 loading & transforms
├── models/            # CNN/backbone definitions
├── losses/            # BCE, Focal, ASL, F‑ECE
├── utils/             # metrics (recall, miss, overflow), helpers
├── train.py           # training entrypoint
├── evaluate.py        # evaluation entrypoint
├── results/           # experiment artifacts (kept) — logs removed
├── assets/            # figures & diagrams (auto-imported if available)
├── requirements.txt
└── LICENSE

```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Training (example)
python train.py --dataset cifar10 --epochs 200 --batch-size 128 --loss fece --lr 0.1

# Evaluation
python evaluate.py --dataset cifar10 --checkpoint path/to/weights.pth
```

## Figure 2 (LFA Visualization)

> **Figure 2** could not be auto-located in the provided archives. Please replace `assets/figure2.png` with the correct image if available.


![Figure 2 placeholder](assets/figure2.png)

## Experimental Tables


### Table 1

*attributes of datasets included in the Original paper of LFA *
| Medical datasets | --- | --- | --- | --- | --- | --- | --- |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| Breast Histopathology  | --- | --- | --- |  CADCAP | --- | --- | --- | 
| Original | | Subset |  |  |  |  |  |
| Condition | Quantity | Condition | Quantity | Condition | Quantity |  | |
| IDC+ | 78,786 | IDC+ | 11,050 | inflammatory | 607 |  |  | 
| IDC- | 198,738 | IDC- | 44,200 | vascularlesions | 605 |  |  | 
|  |  |  |  | healthy control | 600 |  |   | 
|  |  |  |  |  |  |  |  |
| Total | 277,524 | Total | 55,259 | Total |  | 1812 |   |
| General datasets | --- | --- | --- |  ---  | --- | --- | --- |   
| \multicolumn{6}{c|}{CelebA \cite{CelebA}} |  |  |  |  |  |  |  |
| \multicolumn{2}{c}{CIFAR10\_LT \cite{deng2012mnist}} |  |  |  |  |  |  |  |
| Label | Quantity | Label | Quantity | Label | Quantity | Label | Quantity |
| 5 o Clock Shadow | 22,516 | Arched Eyebrows | 54,090 | Attractive | 103,833 | Airplane | 5000 |
| Bags Under Eyes | 41,446 | Bald | 4,547 | Bangs | 30,709 | Automobile | 3,422 |
| Big Lips | 48,785 | Big Nose | 47,516 | Black Hair | 48,472 | Bird | 3,650 |
| Blond Hair | 29,983 | Blurry | 10,312 | Brown Hair | 41,572 | Cat | 3,885 |
| Bushy Eyebrows | 28,803 | Chubby | 11,663 | Double Chin | 9,459 | Deer | 3,020 |
| Eyeglasses | 13,193 | Goatee | 12,716 | Gray Hair | 8,499 | Dog | 228 |
| Heavy Makeup | 78,390 | High Cheekbones | 92,189 | Male | 84,434 | Frog | 2,525 |
| Mouth Slightly Open | 97,942 | Mustache | 8,417 | Narrow Eyes | 23,329 | Horse | 4,308 |
| No Beard | 169,158 | Oval Face | 57,567 | Pale Skin | 8,701 | Ship | 2,259 |
| Pointy Nose | 56,210 | Receding Hairline | 16,163 | Rosy Cheeks | 13,315 | Truck | 4,934 |
| Sideburns | 11,449 | Smiling | 97,669 | Straight Hair | 42,222 |  |  |
| Wavy Hair | 64,744 | Wearing Earrings | 38,276 | Wearing Hat | 9,818 |  |  |
| Wearing Lipstick | 95,715 | Wearing Necklace | 24,913 | Wearing Necktie | 14,732 |  |  |
| Young | 156,734 |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | Total | 202,599 |  |  | Total | 70000 |


### Table 2

*Overall performance of ResNet-50 trained on the ImageNet-LT dataset with different loss functions. Reported values represent mean± standard deviation across all classes.*
| loss | AUC | threshold | acc | Precision | Recall | F0.5 | F0 | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASL | 0.856±0.080 | 0.009±0.031 | 0.978±0.052 | 0.998±0.002 | 0.978±0.052 | 0.993±0.018 | 0.998±0.002 | 0.987±0.033 |
| BCE | 0.863±0.077 | 0.008±0.026 | 0.979±0.043 | 0.998±0.002 | 0.979±0.043 | 0.994±0.012 | 0.998±0.002 | 0.988±0.026 |
| ECE | 0.943±0.038 | 0.017±0.009 | 0.997±0.003 | 0.998±0.001 | 0.997±0.003 | 0.998±0.002 | 0.998±0.001 | 0.998±0.002 |
| F-ECE | 0.943±0.039 | 0.019±0.011 | 0.997±0.003 | 0.998±0.001 | 0.997±0.003 | 0.998±0.002 | 0.998±0.001 | 0.998±0.002 |
| focal | 0.857±0.084 | 0.009±0.027 | 0.982±0.040 | 0.998±0.002 | 0.982±0.040 | 0.994±0.011 | 0.998±0.002 | 0.989±0.024 |


### Table 3

*Quantitative evaluation of models trained on the CAD-CAP dataset~\cite{leenhardt2020cad*
| loss | AUC | threshold | acc | Precision | Recall | F0.5 | F0 | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \multicolumn{9}{c}{DenseNet \cite{huang2017densely}} |  |  |  |  |  |  |  |  |
| ASL | 0.70±0.02 | 0.56±0.05 | 0.58±0.05 | {0.70±0.02} | 0.58±0.05 | 0.63±0.04 | 0.70±0.02 | 0.58±0.06 |
| BCE | 0.70±0.03 | 0.41±0.09 | 0.60±0.04 | 0.69±0.03 | 0.60±0.04 | 0.64±0.02 | 0.69±0.03 | {0.61±0.04} |
| ECE | 0.67±0.03 | 0.5±0.1 | 0.56±0.05 | 0.69±0.04 | 0.56±0.05 | 0.61±0.05 | 0.69±0.04 | 0.56±0.07 |
| F-ECE | {0.71±0.03} | 0.56±0.05 | {0.61±0.04} | 0.69±0.03 | {0.61±0.04} | 0.65±0.03 | 0.69±0.03 | {0.61±0.04} |
| focal | 0.67±0.04 | 0.51±0.09 | 0.57±0.04 | 0.68±0.04 | 0.57±0.04 | 0.62±0.03 | 0.68±0.04 | 0.57±0.05 |
| \multicolumn{9}{c}{ResNet \cite{he2015deep}} |  |  |  |  |  |  |  |  |
| ASL | 0.76±0.04 | 0.53±0.04 | 0.65±0.05 | 0.73±0.03 | 0.65±0.05 | 0.69±0.03 | 0.73±0.03 | 0.65±0.05 |
| BCE | 0.77±0.04 | 0.26±0.04 | 0.65±0.06 | 0.74±0.03 | 0.65±0.06 | 0.70±0.04 | 0.74±0.03 | 0.66±0.06 |
| ECE | 0.78±0.03 | 0.38±0.05 | 0.66±0.07 | {0.75±0.03} | 0.66±0.07 | 0.70±0.04 | 0.75±0.03 | 0.66±0.07 |
| F-ECE | {0.80±0.04} | 0.54±0.05 | {0.70±0.05} | {0.75±0.03} | {0.70±0.05} | 0.73±0.04 | 0.75±0.03 | {0.70±0.05} |
| focal | 0.78±0.04 | 0.41±0.06 | 0.67±0.05 | 0.74±0.03 | 0.67±0.05 | 0.71±0.03 | 0.74±0.03 | 0.68±0.05 |
| \multicolumn{9}{c}{PGCon \cite{vats2022labels}} |  |  |  |  |  |  |  |  |
| ASL | 0.89±0.01 | 0.52±0.07 | 0.81±0.02 | 0.83±0.01 | 0.81±0.02 | 0.82±0.01 | 0.83±0.01 | 0.82±0.02 |
| BCE | 0.81±0.03 | 0.33±0.05 | 0.75±0.02 | 0.77±0.02 | 0.75±0.02 | 0.76±0.02 | 0.77±0.02 | 0.76±0.02 |
| ECE | 0.82±0.03 | 0.37±0.05 | 0.73±0.04 | 0.77±0.02 | 0.73±0.04 | 0.75±0.02 | 0.77±0.02 | 0.74±0.03 |
| F-ECE | {0.94±0.02} | 0.60±0.11 | {0.88±0.02} | {0.88±0.02} | 0.{88±0.02} | 0.88±0.02 | 0.88±0.02 | {0.88±0.02} |
| focal | 0.86±0.03 | 0.42±0.06 | 0.78±0.04 | 0.79±0.03 | 0.78±0.04 | 0.79±0.03 | 0.79±0.03 | 0.78±0.03 |


### Table 4

*Evaluation results of models trained on the Breast Histopathological dataset~\cite{janowczyk2016deep*
| Loss | AUC | Threshold | Acc | Precision | Recall | F0.5 | F0 | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \multicolumn{9}{c}{AlexNet \cite{10.1145/3065386}} |  |  |  |  |  |  |  |  |
| ASL | 0.94±0.00 | 0.60±0.01 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 |
| BCE | 0.94±0.00 | 0.38±0.05 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 |
| ECE | 0.94±0.00 | 0.44±0.02 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 |
| F-ECE | 0.94±0.00 | 0.59±0.02 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 |
| Focal | 0.94±0.03 | 0.43±0.02 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 |
| \multicolumn{9}{c}{BotNet \cite{srinivas2021bottlenecktransformersvisualrecognition}} |  |  |  |  |  |  |  |  |
| ASL | 0.95±0.00 | 0.59±0.03 | 0.89±0.01 | 0.89±0.00 | 0.89±0.01 | 0.89±0.00 | 0.89±0.00 | 0.89±0.00 |
| BCE | 0.94±0.00 | 0.40±0.09 | 0.88±0.00 | 0.89±0.00 | 0.88±0.00 | 0.89±0.00 | 0.89±0.00 | 0.89±0.00 |
| ECE | 0.94±0.00 | 0.43±0.03 | 0.88±0.01 | 0.89±0.01 | 0.88±0.01 | 0.89±0.01 | 0.89±0.01 | 0.88±0.01 |
| F-ECE | 0.94±0.00 | 0.59±0.05 | 0.88±0.00 | 0.89±0.00 | 0.88±0.00 | 0.89±0.00 | 0.89±0.00 | 0.88±0.00 |
| Focal | 0.94±0.01 | 0.45±0.03 | 0.89±0.01 | 0.89±0.01 | 0.89±0.01 | 0.89±0.01 | 0.89±0.01 | 0.89±0.01 |
| \multicolumn{9}{c}{ResNet \cite{he2015deep}} |  |  |  |  |  |  |  |  |
| ASL | 0.94±0.01 | 0.59±0.04 | 0.87±0.01 | 0.88±0.01 | 0.87±0.01 | 0.88±0.01 | 0.88±0.01 | 0.88±0.01 |
| BCE | 0.94±0.00 | 0.39±0.11 | 0.88±0.01 | 0.89±0.01 | 0.88±0.01 | 0.89±0.01 | 0.89±0.01 | 0.89±0.01 |
| ECE | 0.92±0.01 | 0.44±0.03 | 0.85±0.01 | 0.86±0.01 | 0.85±0.01 | 0.86±0.01 | 0.86±0.01 | 0.86±0.01 |
| F-ECE | 0.94±0.00 | 0.60±0.03 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 | 0.88±0.00 |
| Focal | 0.94±0.01 | 0.46±0.05 | 0.88±0.01 | 0.88±0.01 | 0.88±0.01 | 0.88±0.01 | 0.88±0.01 | 0.88±0.01 |


### Table 5

*Evaluation results of models trained on the Breast Histopathological dataset~\cite{janowczyk2016deep*
| Ratio | Loss | AUC | Threshold | Acc | Precision | Recall | F0.5 | F0 | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \multirow{5}{*}{50 (1:4)} |  |  |  |  |  |  |  |  |  |
|  | ASL | 0.90±0.01 | 0.57±0.02 | 0.86±0.01 | 0.87±0.00 | 0.86±0.01 | 0.87±0.01 | 0.87±0.00 | 0.86±0.01 |
|  | BCE | 0.90±0.01 | 0.33±0.07 | 0.86±0.01 | 0.87±0.01 | 0.86±0.01 | 0.87±0.01 | 0.87±0.01 | 0.86±0.01 |
|  | ECE | 0.88±0.01 | 0.50±0.10 | 0.84±0.01 | 0.86±0.01 | 0.84±0.01 | 0.85±0.01 | 0.86±0.01 | 0.85±0.01 |
|  | F-ECE | 0.90±0.01 | 0.59±0.01 | 0.85±0.02 | 0.86±0.01 | 0.85±0.02 | 0.86±0.01 | 0.86±0.01 | 0.86±0.01 |
|  | Focal | 0.90±0.02 | 0.43±0.03 | 0.86±0.01 | 0.87±0.01 | 0.86±0.01 | 0.87±0.01 | 0.87±0.01 | 0.86±0.01 |
| \multirow{5}{*}{25 (1:8)} |  |  |  |  |  |  |  |  |  |
|  | ASL | 0.87±0.02 | 0.53±0.01 | 0.84±0.01 | 0.86±0.01 | 0.84±0.01 | 0.85±0.01 | 0.86±0.01 | 0.85±0.01 |
|  | BCE | 0.88±0.02 | 0.19±0.02 | 0.84±0.01 | 0.86±0.01 | 0.84±0.01 | 0.85±0.01 | 0.86±0.01 | 0.84±0.01 |
|  | ECE | 0.87±0.02 | 0.33±0.02 | 0.84±0.01 | 0.85±0.01 | 0.84±0.01 | 0.85±0.01 | 0.85±0.01 | 0.85±0.01 |
|  | F-ECE | 0.87±0.02 | 0.52±0.04 | 0.84±0.02 | 0.85±0.01 | 0.84±0.02 | 0.85±0.01 | 0.85±0.01 | 0.85±0.01 |
|  | Focal | 0.87±0.01 | 0.40±0.01 | 0.84±0.01 | 0.86±0.01 | 0.84±0.01 | 0.85±0.01 | 0.86±0.01 | 0.85±0.01 |
| \multirow{5}{*}{12 (3:50)} |  |  |  |  |  |  |  |  |  |
|  | ASL | 0.86±0.02 | 0.49±0.01 | 0.84±0.01 | 0.85±0.01 | 0.84±0.01 | 0.85±0.01 | 0.85±0.01 | 0.84±0.01 |
|  | BCE | 0.87±0.02 | 0.15±0.02 | 0.85±0.01 | 0.86±0.01 | 0.85±0.01 | 0.85±0.01 | 0.86±0.01 | 0.85±0.01 |
|  | ECE | 0.86±0.02 | 0.32±0.03 | 0.83±0.02 | 0.85±0.01 | 0.83±0.02 | 0.84±0.01 | 0.85±0.01 | 0.84±0.01 |
|  | F-ECE | 0.87±0.01 | 0.52±0.04 | 0.84±0.01 | 0.85±0.01 | 0.84±0.01 | 0.85±0.01 | 0.85±0.01 | 0.84±0.01 |
|  | Focal | 0.86±0.02 | 0.33±0.02 | 0.83±0.01 | 0.85±0.01 | 0.83±0.01 | 0.84±0.01 | 0.85±0.01 | 0.84±0.01 |
| \multirow{5}{*}{6 (3:100)} |  |  |  |  |  |  |  |  |  |
|  | ASL | 0.84±0.02 | 0.52±0.03 | 0.82±0.02 | 0.83±0.01 | 0.82±0.02 | 0.83±0.01 | 0.83±0.01 | 0.83±0.01 |
|  | BCE | 0.85±0.01 | 0.20±0.06 | 0.84±0.01 | 0.85±0.01 | 0.84±0.01 | 0.84±0.01 | 0.85±0.01 | 0.84±0.01 |
|  | ECE | 0.83±0.01 | 0.33±0.04 | 0.81±0.02 | 0.83±0.01 | 0.81±0.02 | 0.83±0.01 | 0.83±0.01 | 0.82±0.01 |
|  | F-ECE | 0.86±0.01 | 0.53±0.07 | 0.84±0.01 | 0.85±0.01 | 0.84±0.01 | 0.85±0.01 | 0.85±0.01 | 0.84±0.01 |
|  | Focal | 0.84±0.02 | 0.40±0.04 | 0.83±0.00 | 0.84±0.01 | 0.83±0.00 | 0.84±0.01 | 0.84±0.01 | 0.84±0.01 |
| \multirow{5}{*}{3 (3:200)} |  |  |  |  |  |  |  |  |  |
|  | ASL | 0.82±0.02 | 0.52±0.02 | 0.82±0.02 | 0.83±0.01 | 0.82±0.02 | 0.83±0.01 | 0.83±0.01 | 0.82±0.02 |
|  | BCE | 0.82±0.01 | 0.20±0.04 | 0.81±0.02 | 0.83±0.00 | 0.81±0.02 | 0.82±0.01 | 0.83±0.00 | 0.82±0.01 |
|  | ECE | 0.79±0.06 | 0.27±0.03 | 0.78±0.05 | 0.81±0.03 | 0.78±0.05 | 0.81±0.04 | 0.81±0.03 | 0.79±0.04 |
|  | F-ECE | 0.82±0.04 | 0.52±0.03 | 0.82±0.03 | 0.82±0.03 | 0.82±0.03 | 0.82±0.03 | 0.82±0.03 | 0.82±0.03 |
|  | Focal | 0.82±0.03 | 0.37±0.05 | 0.80±0.03 | 0.82±0.02 | 0.80±0.03 | 0.81±0.02 | 0.82±0.02 | 0.80±0.02 |
| \multirow{5}{*}{1 (1:200)} |  |  |  |  |  |  |  |  |  |
|  | ASL | 0.72±0.13 | 0.50±0.07 | 0.66±0.23 | 0.65±0.31 | 0.66±0.23 | 0.65±0.30 | 0.65±0.31 | 0.65±0.29 |
|  | BCE | 0.77±0.02 | 0.18±0.06 | 0.79±0.03 | 0.81±0.01 | 0.79±0.03 | 0.80±0.02 | 0.81±0.01 | 0.79±0.03 |
|  | ECE | 0.62±0.20 | 0.31±0.08 | 0.57±0.27 | 0.79±0.03 | 0.57±0.27 | 0.57±0.31 | 0.79±0.03 | 0.54±0.32 |
|  | F-ECE | 0.79±0.08 | 0.47±0.09 | 0.79±0.06 | 0.81±0.04 | 0.79±0.06 | 0.80±0.04 | 0.81±0.04 | 0.80±0.06 |
|  | Focal | 0.78±0.03 | 0.37±0.05 | 0.78±0.03 | 0.81±0.02 | 0.78±0.03 | 0.80±0.02 | 0.81±0.02 | 0.79±0.03 |


## Reproducibility Notes


- All experiment **logs** and temporary run artifacts have been **removed**; final **results** (e.g., CSVs) are preserved under `results/`.

- A conservative static pass identified top‑level functions/classes with **no references** across the codebase and moved them to `utils/_archive_unused.py` for safety. This keeps the working set minimal while preserving code history.

- If any removal affected your custom pipeline, simply restore items from the archive file.


## Citation


If you use this repository or F‑ECE in your work, please cite the corresponding paper (update with your final BibTeX):
```bibtex
@article{your2025lfa,
  title   = {Loss Function Assessment and Focal-balanced Exponential Cross-Entropy for Hard-to-Learn Classes},
  author  = {Liu, Bozhao and Wang, Jin and Others},
  journal = {Under Review},
  year    = {2025}
}
```
