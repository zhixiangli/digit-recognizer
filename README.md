# digit-recognizer

Use CNN to achieve 99% in **[digit recognizer](https://www.kaggle.com/c/digit-recognizer)**

## Download Data

Download training & testing data from [kaggle](https://www.kaggle.com/c/digit-recognizer/data)

## Setup

Requires **Python 3.12+**.

Create a virtual environment and install dependencies:

```
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install .
```

To upgrade dependencies to their latest compatible versions:

```
pip install --upgrade .
```

## Quick Start

**Command Format**:

```
python cnn.py (training data path) (testing data path) (predict output path)
```

**Example**:

```
python cnn.py ./train.csv ./test.csv ./res.csv
```