from typing import Iterable

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.decomposition import PCA  # type: ignore

from src.definitions import PATH_TO_HOUSING_DATA
from src.main.stochastic_regression_imputation import stochastic_regression_imputer


def main():
    feature_to_impute = "LotFrontage"

    df = pd.read_csv(PATH_TO_HOUSING_DATA)

    df = drop_non_numeric_features(df)
    df = drop_other_features_with_missing_data(df, feature_to_impute=feature_to_impute)

    print(f"dataset has {len(df)} rows and and {len(df.columns)} features")
    print(f"LostFrontage has {df.LotFrontage.isnull().sum()} missing datapoints")

    print(f"Running stochastic regression imputation")
    df = stochastic_regression_imputer(df, "LotFrontage")

    print(f"dataset has {len(df)} rows and and {len(df.columns)} features")
    print(f"LostFrontage has {df.LotFrontage.isnull().sum()} missing datapoints")

    render_imputed_data(df)


def drop_non_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Only keeping numerical features")
    return df[determine_numeric_features(df)]


def drop_other_features_with_missing_data(
    df: pd.DataFrame, feature_to_impute: str
) -> pd.DataFrame:
    print(f"Only keeping features without missing data (and '{feature_to_impute}')")
    features_with_missing_data = determine_features_with_missing_data(df, threshold=0.0)
    other_features_with_missing_data = set(features_with_missing_data)
    other_features_with_missing_data.remove(feature_to_impute)
    return df.drop(columns=other_features_with_missing_data)


def determine_numeric_features(df: pd.DataFrame) -> pd.Series:
    return df.select_dtypes(include={"int64", "float64"}).columns


def determine_features_with_missing_data(
    df: pd.DataFrame, threshold: float
) -> Iterable[str]:
    return {
        feature
        for feature in df.columns
        if df[feature].isnull().sum() / len(df) > threshold
    }


def render_imputed_data(df: pd.DataFrame) -> None:
    pca: PCA = PCA(n_components=1)
    excluded_columns = ["LotFrontage", "LotFrontage_missing"]
    pca.fit(df.drop(columns=excluded_columns))
    plt.scatter(
        x=pca.transform(df.drop(columns=excluded_columns)),
        y=df["LotFrontage"],
        c=df["LotFrontage_missing"],
    )
    plt.show()


if __name__ == "__main__":
    main()
