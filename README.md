# Titanic survival prediction

Two Docker containers fit logistic regression models to the Titanic passenger manifest. Each prints its data checks and accuracy to the terminal, then writes a prediction for every passenger in the competition test set.

## 1. Clone the repository

Install [Git](https://git-scm.com/downloads) and [Docker with Compose v2](https://docs.docker.com/compose/install/). Start Docker before running the commands. On Windows, Docker Desktop should use Linux containers. These commands work in PowerShell, macOS Terminal, and Linux shells.

```sh
git clone https://github.com/li13232/li13232-li13232-titanic-survival-docker.git
cd li13232-li13232-titanic-survival-docker
docker compose version
```

## 2. Download the passenger data

1. Sign in to Kaggle and open the [Titanic competition data page](https://www.kaggle.com/competitions/titanic/data). If prompted, join the competition to enable downloads.
2. Choose **Download All**, extract the archive, and copy the two files below directly into the existing `src/data/` folder. Keep the original headers and filenames.

```text
src/data/train.csv    891 passengers, includes Survived
src/data/test.csv     418 passengers, includes PassengerId
```

The archive also includes `gender_submission.csv`, a sample submission based on passenger sex. Only `train.csv` and `test.csv` are needed for the standard run. Data and predictions are excluded from Git.

## 3. Build and run both models

From the repository root, run these commands in order. Docker downloads the two base images and installs the pinned dependencies on the first build.

```sh
docker compose build
docker compose run --rm python
docker compose run --rm r
```

The services use the Dockerfiles in `src/python/` and `src/r/`. Compose mounts `src/data/` read-only and creates `predictions/` on the host. To rerun a model, repeat its `run` command. Rebuild after changing a script or dependency.

| Run | Prediction file | Rows | Columns |
| --- | --- | ---: | --- |
| Python | `predictions/python_predictions.csv` | 418 | `PassengerId,Survived` |
| R | `predictions/r_predictions.csv` | 418 | `PassengerId,Survived` |

The rows follow the order of `test.csv`. A prediction of 1 means survived and 0 means did not survive.

## What the terminal reports

Both scripts print row counts, survival rates by passenger class, missing-value counts, preprocessing steps, and the number of predicted survivors. Of the 891 training passengers, 342 survived. Survival ranged from 62.96% in first class (136/216) to 24.24% in third class (119/491). Age is missing for 177 training passengers, so both models retain those rows and impute age.

Both models use passenger class (ordinal), sex, age, siblings/spouses aboard, parents/children aboard, and fare. Missing numeric values use the fitting sample's medians and missing sex uses its most frequent value. Python standardizes numeric inputs, one-hot encodes sex, and fits scikit-learn's regularized logistic regression. R encodes sex as a binary indicator and fits a binomial model with `stats::glm`.

Each script holds out roughly 20% of the labeled training records, stratified by survival with seed 42. It fits preprocessing and the model on the remaining rows, evaluates on the holdout, then refits on all 891 passengers before predicting the competition test set.

| Language | Held-out validation accuracy | Full training accuracy | Predicted survivors in test.csv |
| --- | ---: | ---: | ---: |
| Python | 144/179 (80.45%) | 709/891 (79.57%) | 160/418 |
| R | 148/178 (83.15%) | 708/891 (79.46%) | 160/418 |

These are results on the standard competition download. The two languages draw different holdout samples, so their validation percentages should be interpreted separately.

## Score the competition test predictions

Kaggle withholds the `Survived` column in its public `test.csv`. To obtain a leaderboard accuracy, choose **Submit Predictions** on the [competition page](https://www.kaggle.com/competitions/titanic), upload either generated CSV, and read the score under **My Submissions**. Each submission scores one model. The [evaluation page](https://www.kaggle.com/competitions/titanic/overview/evaluation) defines accuracy as the fraction of passengers classified correctly.

If an instructor provides ground truth, save it as `src/data/test_labels.csv` with columns `PassengerId,Survived`, one binary outcome for every test ID. Rerun both services to print exact labeled-test accuracy. The scripts match outcomes by ID even when the rows are shuffled. Kaggle's `gender_submission.csv` contains baseline predictions and cannot supply ground truth.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Docker cannot connect to the daemon | Start Docker Desktop or the Docker Engine, then retry `docker compose build`. |
| The script cannot find `train.csv` or `test.csv` | Check that the extracted files are directly inside `src/data/` and that you ran Compose from the repository root. |
| A run says test accuracy is unavailable | This is expected with the standard Kaggle files. The prediction CSV is still written to `predictions/`. |
