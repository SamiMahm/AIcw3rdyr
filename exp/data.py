import pandas as pd

from config import DATA_PATH


def load_raw() -> pd.DataFrame:

    df = pd.read_csv(DATA_PATH, dtype={"id": str, "zipcode": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
    df["zipcode"] = df["zipcode"].astype(int)
    df["floors"] = df["floors"].astype(float)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["house_age"] = df["date"].dt.year - df["yr_built"]
    df["renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["price_per_sqft"] = df["price"] / df["sqft_living"]
    return df


def load() -> pd.DataFrame:
    df = add_derived_features(load_raw())
    return df.sort_values("date").reset_index(drop=True)
