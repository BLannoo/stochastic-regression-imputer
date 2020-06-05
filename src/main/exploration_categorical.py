import pandas as pd  # type: ignore


def general_stats(df: pd.DataFrame):
    print(f"There are {len(df)} rows and {len(df.columns)} columns")
    print()


def remove_features_without_data(
    df: pd.DataFrame, inplace: bool = False
) -> pd.DataFrame:
    columns_without_data = [column for column in df.columns if df[column].count() == 0]

    print(f"The following columns contain no data:\n\t{columns_without_data}")
    print()

    return df.drop(columns=columns_without_data, inplace=inplace)


def remove_features_with_fixed_data(
    df: pd.DataFrame, inplace: bool = False
) -> pd.DataFrame:
    fixed_features = {
        column: df[column].iloc[0]
        for column in df.columns
        if len(df[column].value_counts()) == 1
    }

    print("The following columns contain a fixed value:")
    for column, value in fixed_features.items():
        print(f'\t"{column}" is always: {value}')
    print()

    return df.drop(
        columns=[column for column, _ in fixed_features.items()], inplace=inplace
    )


def print_features_with_limited_variaty(df: pd.DataFrame, threshold: int = 10) -> None:
    categorical_features = {
        column: df[column].value_counts()
        for column in df.columns
        if len(df[column].value_counts()) <= threshold
    }

    print(
        f"The following columns contain a limited (<= {threshold}) variaty of values:"
    )
    for column, counts in categorical_features.items():
        print(f'\t"{column}": ')
        for value, count in counts.iteritems():
            print(f'\t\t(#={count:7}):\t"{value}"')
    print()


def basic_pre_processing(df: pd.DataFrame) -> pd.DataFrame:
    general_stats(df)
    df = remove_features_without_data(df)
    df = remove_features_with_fixed_data(df)
    print_features_with_limited_variaty(df)
    return df
