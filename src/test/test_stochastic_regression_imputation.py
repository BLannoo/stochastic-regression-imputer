import random
from typing import Callable, TypeVar

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from assertpy import assert_that  # type: ignore

from src.main.stochastic_regression_imputation import stochastic_regression_imputer

FLOATS = TypeVar("FLOATS", np.ndarray, pd.Series)


def test_stochastic_regression_imputation():
    np.random.seed(0)

    # Given a fake dataset with a known relation and error around that relation
    def relation(x: FLOATS) -> FLOATS:
        return 2.0 * x + 10.0

    error_std = 0.3
    size_dataset = 1_000
    missing_data_fraction = 0.1

    df = generate_fake_dataset(size_dataset, relation, missing_data_fraction, error_std)

    # When imputing the missing data
    df_imputed = stochastic_regression_imputer(df, feature="y")

    # Then no data is lost
    assert_that(df_imputed).is_length(len(df))

    # Then a correct number of rows are labeled as missing
    assert_that(df_imputed["y_missing"].sum()).is_close_to(
        size_dataset * missing_data_fraction, tolerance=0.5
    )

    # Then the errors around the relation are maintained
    actually_imputed = df_imputed[df_imputed["y_missing"]]
    error_after_imputation = relation(actually_imputed["x"]) - actually_imputed["y"]
    assert_that(error_after_imputation.mean()).is_close_to(0.0, tolerance=0.1)
    assert_that(error_after_imputation.std()).is_close_to(error_std, tolerance=0.1)

    # Then the imputed data seems spread out similar to the original data
    render_imputed_data(df_imputed)


def generate_fake_dataset(
    size_dataset: int,
    relation: Callable[[FLOATS], FLOATS],
    missing_data_fraction: float,
    error_std: float,
) -> pd.DataFrame:
    x = np.random.normal(size=size_dataset)
    y = relation(x) + np.random.normal(scale=error_std, size=size_dataset)
    df = pd.DataFrame(data={"x": x, "y": y})

    sample = random.sample(
        range(size_dataset), int(size_dataset * missing_data_fraction)
    )
    df["y"][sample] = np.NaN

    return df


def render_imputed_data(df_imputed: pd.DataFrame) -> None:
    plt.scatter(
        x=df_imputed["x"], y=df_imputed["y"], c=df_imputed["y_missing"],
    )
    plt.title(
        "Confirm visually that the imputed data is spread out similar to the original data"
    )
    plt.show()
