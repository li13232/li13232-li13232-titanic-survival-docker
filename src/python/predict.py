"""Fit a Titanic survival model and produce a Kaggle-format prediction file."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
FEATURES = [*NUMERIC, "Sex"]


def load_csv(path, required):
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    ids = frame["PassengerId"]
    if frame.empty or ids.isna().any() or not ids.is_unique:
        raise ValueError(f"{path.name} must have one row per distinct, nonempty PassengerId.")
    if "Survived" in frame and not frame["Survived"].isin([0, 1]).all():
        raise ValueError(f"{path.name}: Survived must contain only 0 or 1.")
    return frame


def print_accuracy(label, truth, prediction):
    correct = int((truth.to_numpy() == prediction).sum())
    print(f"{label}: {correct}/{len(truth)} ({correct / len(truth):.2%})")


def test_outcomes(test, path):
    if path.exists():
        labels = load_csv(path, ["PassengerId", "Survived"]).set_index("PassengerId")
        if set(labels.index) != set(test["PassengerId"]):
            raise ValueError("test_labels.csv must cover exactly the PassengerIds in test.csv.")
        matched = labels.loc[test["PassengerId"], "Survived"].reset_index(drop=True)
        if "Survived" in test and not (matched == test["Survived"]).all():
            raise ValueError("The outcomes in test.csv and test_labels.csv disagree.")
        return matched
    return test.get("Survived")


def run(data_dir, output_dir):
    train = load_csv(data_dir / "train.csv", ["PassengerId", "Survived", *FEATURES])
    test = load_csv(data_dir / "test.csv", ["PassengerId", *FEATURES])
    truth = test_outcomes(test, data_dir / "test_labels.csv")
    print(f"Loaded {len(train)} training passengers and {len(test)} test passengers.")
    print(f"Training survivors: {train.Survived.sum()}/{len(train)} ({train.Survived.mean():.1%})")
    for column in ["Sex", "Pclass"]:
        print(f"Training survival by {column} (survivors / passengers):")
        for group, rows in train.groupby(column, dropna=False):
            outcomes = rows["Survived"]
            print(f"  {group}: {outcomes.sum()}/{len(rows)} ({outcomes.mean():.1%})")

    print("Model features:", ", ".join(FEATURES))
    print("Missing values in model features:")
    print(pd.DataFrame({"train": train[FEATURES].isna().sum(),
                        "test": test[FEATURES].isna().sum()}).to_string())
    print("Numeric features: impute medians and standardize; Pclass is ordinal.")
    print("Sex: impute the most common value and one-hot encode.")
    transform = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("sex", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), ["Sex"]),
    ])
    model = make_pipeline(transform, LogisticRegression(max_iter=1000))
    fit, validation = train_test_split(
        train, test_size=0.20, stratify=train["Survived"], random_state=42
    )
    print(f"Stratified split (seed 42): {len(fit)} fit, {len(validation)} held out.")
    print("Fitting preprocessing and logistic regression on the fit partition.")
    model.fit(fit[FEATURES], fit["Survived"])
    print_accuracy("Held-out validation accuracy", validation["Survived"],
                   model.predict(validation[FEATURES]))

    print(f"Refitting preprocessing and logistic regression on all {len(train)} training passengers.")
    model.fit(train[FEATURES], train["Survived"])
    print_accuracy("Full training accuracy", train["Survived"], model.predict(train[FEATURES]))
    predictions = model.predict(test[FEATURES]).astype(int)
    result = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
    if truth is not None:
        print_accuracy("Labeled test accuracy", truth, predictions)
    else:
        print("Kaggle test accuracy: unavailable locally because test.csv has no survival labels.")
        print("Submit the predictions on Kaggle for its leaderboard score.")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "python_predictions.csv"
    result.to_csv(target, index=False)
    print(f"Wrote {len(result)} predictions to {target}; predicted survivors: {predictions.sum()}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("/data"))
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("/output"))
    args = parser.parse_args()
    try:
        run(args.data_dir, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
