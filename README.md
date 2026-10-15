# Plant Disease Classification

Comparison of a baseline CNN and an ImageNet-pretrained MobileNetV2 (transfer learning, frozen backbone) for 23-class plant leaf disease classification. Built with TensorFlow/Keras as a course project (2026.05).

## Dataset
- Kaggle "Plant Disease Detection" (karagwaanntreasure): https://www.kaggle.com/datasets/karagwaanntreasure/plant-disease-detection
- 23 classes, 5 crops (apple, corn, pepper, potato, tomato); 35,725 images; folder-per-class layout.
- The dataset page notes that classes may be imbalanced. Original image source: [not stated on the page].

## Split
- `data_Separation.py`: class-wise random 70/20/10 split -> 24,997 train / 7,136 val / 3,592 test.
- Shuffle is **not seeded**, and the split was run **once**; all experiments below use the same split.
- Exact-duplicate check (MD5): 1 identical file found across train/test (0.03% of the test set). Near-duplicates were not checked.

## Setup (all runs)
- Input 128x128, augmentation (rotation 20, zoom 0.2, horizontal flip) on train only, pixel scaling 1/255.
- Adam, categorical cross-entropy, **5 epochs** (limited by class time), single run per configuration.
- Basic CNN: 2 conv blocks + Dense(128) + Dropout(0.5), 7.4M trainable params.
- MobileNetV2 (default): ImageNet weights, backbone frozen, GlobalAveragePooling + Dense(128) + Dropout(0.5) + softmax; 167K trainable params, lr 1e-3, batch 16.

## Results (test set: 3,592 images; 95% CI about +/-1.1 pp)
| Model / configuration | Script | Val acc | Test acc |
|---|---|---|---|
| Basic CNN | `Basic_CNN.py` | 82.51% | **82.79%** |
| MobileNetV2 (default: lr 1e-3, Dense 128, batch 16) | `MobileNetV2.py` | 88.55% | **87.53%** |
| lr 1e-4 | `MobileNetV2_lr` | 87.46% | 86.83% |
| lr 1e-2 | `MobileNetV2_lr` | 57.47% | 56.15% |
| Dense 64 | `MobileNetV2_Dense` | 86.53% | 86.05% |
| Dense 256 | `MobileNetV2_Dense` | 89.34% | 89.23% |
| batch 8 | `MobileNetV2_batch_size` | 86.95% | 86.61% |
| batch 32 | `MobileNetV2_batch_size` | 88.93% | 88.42% |

Learning curves and logs: `results/`.

## Observations
- Transfer learning (frozen backbone, 167K trainable params) outperformed the baseline CNN (7.4M params) by 4.7 pp on test (single runs).
- Hidden-layer width 64 -> 128 -> 256 increased accuracy monotonically (86.1 -> 87.5 -> 89.2%); batch-size and lr 1e-4 differences are within sampling error. lr 1e-2 failed to train.
- Headline numbers use the default configuration. Variants were all evaluated on the same test set, so the best variant is not an independent estimate.
- Training/validation curves were still improving at epoch 5 (underfit).

## Limitations / not explored
- Single runs, no fixed seeds; split not seeded.
- Model-specific preprocessing (`preprocess_input`), backbone fine-tuning, per-class metrics (macro-F1, confusion matrix), and class-imbalance handling were not tried.

## Files
`data_Separation.py` (split), `data_Count.py` (class counts), `Basic_CNN.py`, `MobileNetV2.py`, `MobileNetV2_lr`, MobileNetV2_Dense, MobileNetV2_batch_size (hyperparameter variants; the active setting is the uncommented line).
