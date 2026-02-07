# digit-recognizer

Use CNN to achieve 99% in **[digit recognizer](https://www.kaggle.com/c/digit-recognizer)**

## Download Data

Download training & testing data from [kaggle](https://www.kaggle.com/c/digit-recognizer/data)

## Setup

Requires **Python 3.12+** and [uv](https://docs.astral.sh/uv/).

Install dependencies:

```
uv sync
```

To upgrade dependencies to their latest compatible versions:

```
uv lock --upgrade
uv sync
```

## Quick Start

**Command Format**:

```
uv run digit-recognizer (training data path) (testing data path) (predict output path)
```

**Example**:

```
uv run digit-recognizer ./train.csv ./test.csv ./res.csv
```