# Flanker fMRI analysis and deep learning classifier

## 🚀 About the project

fMRI is an advanced technique for capturing brain functionality. There are plenty of methods to analyze fMRI data, such as FSL, SPM, and AFNI. People can recognize disabilities and brain activity through this analysis, but can a computer do it for us?

In this project, I analyzed fMRI images from the Flanker dataset. Based on the analysis, I built deep learning-based classifier models to distinguish whether the brain is performing a "congruent" task or an "incongruent" task. Due to the small size of the dataset and the presence of noise, the performance was not very good. Therefore, I suggest several alternative approaches, such as a multiview model and a soft voting model. I believe these methods will become more robust with a larger dataset.

## fMRI analysis

### Flanker dataset

### Preprocessing

### Analysis

## Deep Learning Classification

### 📁 Project structure 🏗️
```
.
├── src/
│   ├── data/
│   │   ├── samples/
│   │   └── labels.csv
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_classifier.py
│   │   ├── multiview_classifier.py
│   │   └── model_select.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── learning.py
│   │   └── metrics.py
│   ├── test.py
│   └── train.py
├── README.md
└── requirements.txt
```

- src
  - data 
    - you can find fMRI image files and label csv files
  - models
    - you can find model classes
  - utils
    - you can find useful functions
  - train.py
  - test.py
- requirements.txt
  - necessary packages to run the codes

