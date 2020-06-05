from typing import Callable, List, TypeVar, Dict, Tuple, NamedTuple, Any

import pandas as pd  # type: ignore

from src.definitions import PATH_TO_HOUSING_DATA

CATEGORICAL_THRESHOLD = 50
MISSING_DATA_THRESHOLD_LOW = 0.1
MISSING_DATA_THRESHOLD_HIGH = 0.9
DOMINANT_VALUE_THRESHOLD = 0.9


def main():
    df = pd.read_csv(PATH_TO_HOUSING_DATA)

    print(f"Dataset has {len(df)} datapoints and {len(df.columns)} features")

    int_features, object_features = analyse_features_by_type(df)

    analyse_categorical_features(df, int_features, object_features)

    analyse_missing_data(df)

    analyse_dominant_values(df)


def analyse_features_by_type(df):
    print(f"The following datatypes are present: {df.dtypes.value_counts().to_dict()}")
    int_features = features_of_type(df, type_name="int64")
    float_features = features_of_type(df, type_name="float64")
    object_features = features_of_type(df, type_name="object")
    # print(f"The float features are: {int_features}")
    print(f"The float features are: {float_features}")
    # print(f"The float features are: {object_features}")
    return int_features, object_features


def analyse_categorical_features(df, int_features, object_features):
    categorical_features = map_as_dict(
        iterable=df.columns,
        function=number_of_different_values_evaluator(df),
        condition=lambda count: count < CATEGORICAL_THRESHOLD,
    )
    # print(f"The categorical features are: {categorical_features}")
    non_categorical_object_features = set(object_features).difference(
        categorical_features.keys()
    )
    print(
        f"{len(non_categorical_object_features)} features are of type object, "
        f"but have more then {CATEGORICAL_THRESHOLD} different values"
    )
    categorical_int_features = set(int_features).intersection(
        categorical_features.keys()
    )
    print(
        f"{len(categorical_int_features)} features might be fake ints (categorical): {categorical_int_features}"
    )


def analyse_missing_data(df):
    features_with_miss_rate = map_as_dict(
        iterable=df.columns,
        function=fraction_of_missing_values_evaluator(df),
        condition=lambda missing_fraction: 0.0 < missing_fraction,
    )
    report_missing_data(features_with_miss_rate)


def analyse_dominant_values(df):
    features_with_dominant_value = map_as_dict(
        iterable=df.columns,
        function=fraction_of_dominant_values_evaluator(df),
        condition=lambda mode: mode.fraction > DOMINANT_VALUE_THRESHOLD,
    )
    print("\n===== Dominant data =====")
    print(
        f"{len(features_with_dominant_value)} features have a value, which covers more than {DOMINANT_VALUE_THRESHOLD} of the cases:\n"
        f"{features_with_dominant_value}"
    )


def features_of_type(df: pd.DataFrame, type_name: str) -> List[str]:
    return list(df.dtypes[df.dtypes == type_name].index)


def number_of_different_values_evaluator(df: pd.DataFrame) -> Callable[[str], int]:
    return lambda feature: len(df[feature].value_counts())


def fraction_of_missing_values_evaluator(df: pd.DataFrame) -> Callable[[str], float]:
    dataset_size = len(df)
    return lambda feature: df[feature].isnull().sum() / dataset_size


class Mode(NamedTuple):
    value: Any
    fraction: float


def fraction_of_dominant_values_evaluator(
    df: pd.DataFrame,
) -> Callable[[str], Tuple[str, float]]:
    dataset_size = len(df)

    def func(feature: str) -> Tuple[str, float]:
        counts = df[feature].value_counts()
        return Mode(value=counts.index[0], fraction=counts.iloc[0] / dataset_size)

    return func


T = TypeVar("T")


def map_as_dict(
    iterable,
    function: Callable[[str], T],
    condition: Callable[[T], bool] = lambda _: True,
) -> Dict[str, T]:
    return {
        feature: value
        for feature, value in map(
            lambda feature: (feature, function(feature)), iterable
        )
        if condition(value)
    }


def report_missing_data(features_with_miss_rate: Dict[str, float]) -> None:
    print("\n===== Missing data =====")
    features_with_high_miss_rate = {
        feature: missing_fraction
        for feature, missing_fraction in features_with_miss_rate.items()
        if MISSING_DATA_THRESHOLD_HIGH <= missing_fraction
    }
    print(
        f"{len(features_with_high_miss_rate)} features with high missing data "
        f"({MISSING_DATA_THRESHOLD_HIGH}<=): {features_with_high_miss_rate}"
    )
    features_with_medium_miss_rate = {
        feature: missing_fraction
        for feature, missing_fraction in features_with_miss_rate.items()
        if MISSING_DATA_THRESHOLD_LOW <= missing_fraction < MISSING_DATA_THRESHOLD_HIGH
    }
    print(
        f"{len(features_with_medium_miss_rate)} features with medium missing data "
        f"({MISSING_DATA_THRESHOLD_LOW}<=x<{MISSING_DATA_THRESHOLD_HIGH}): {features_with_medium_miss_rate}"
    )
    features_with_low_miss_rate = {
        feature: missing_fraction
        for feature, missing_fraction in features_with_miss_rate.items()
        if missing_fraction < MISSING_DATA_THRESHOLD_LOW
    }
    print(
        f"{len(features_with_low_miss_rate)} features with low missing data"
        f" (0.0<x<{MISSING_DATA_THRESHOLD_LOW}): {features_with_low_miss_rate}"
    )


if __name__ == "__main__":
    main()
