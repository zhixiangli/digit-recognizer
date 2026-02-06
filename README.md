# digit-recognizer

Use CNN to achieve 99% in **[digit recognizer](https://www.kaggle.com/c/digit-recognizer)**

## Prerequisites

- Python >= 3.10
- Dependencies listed in `requirements.txt` / `pyproject.toml`

## Setup

```
pip install -r requirements.txt
```

## Download Data

Download training & testing data from [kaggle](https://www.kaggle.com/c/digit-recognizer/data)

## Quick Start

**Command Format**:

```
python cnn.py (training data path) (testing data path) (predict output path)
```

**Example**:

```
python cnn.py ./train.csv ./test.csv ./res.csv
```