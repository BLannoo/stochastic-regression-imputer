import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore


def stochastic_regression_imputer(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Replace missing values for 'feature' by the sum of
    1) a linear regression prediction
    2) an error with appropriate distribution
    """

    df[feature + "_missing"] = False
    df.loc[df[feature].isnull(), feature + "_missing"] = True

    df_with_feature = df[~df[feature].isnull()]
    regression = LinearRegression().fit(
        X=df_with_feature.drop(columns=[feature]), y=df_with_feature[feature]
    )

    error = (
        regression.predict(X=df_with_feature.drop(columns=[feature]))
        - df_with_feature[feature]
    )

    standard_deviation = error.std()

    df_without_feature = df[df[feature].isnull()].drop(columns=[feature])
    df_without_feature[feature] = regression.predict(
        X=df_without_feature
    ) + np.random.normal(
        loc=0.0, scale=standard_deviation, size=len(df_without_feature)
    )

    result = pd.concat([df_with_feature, df_without_feature])

    return result
