
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit, train_test_split

TEST_SIZE = 0.20
N_AREAS = 25  # number of spatial clusters (areas) for the spatial split


def rand_split(df, seed):
    idx = np.arange(len(df))
    return train_test_split(idx, test_size=TEST_SIZE, random_state=seed)


def house_split(df, seed):
    # GroupShuffleSplit picks ~20% of the GROUPS (houses), not rows
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=seed)
    return next(gss.split(df, groups=df["id"]))


def temp_split(df, seed=None):
    # seed unused: the cut-off date is fixed. Kept in the signature so
    # all four functions can be called the same way.
    order = np.argsort(df["date"].values, kind="stable")
    cut = int(len(df) * (1 - TEST_SIZE))
    return np.sort(order[:cut]), np.sort(order[cut:])


def spatial_areas(df, seed):

    xy = np.column_stack([df["lat"], df["long"] * np.cos(np.radians(47.5))])
    return KMeans(n_clusters=N_AREAS, n_init=10,
                  random_state=seed).fit_predict(xy)


def spatial_split(df, seed):
    #Hold out whole areas until they contain 20% of all sales
    areas = spatial_areas(df, seed)
    sizes = pd.Series(areas).value_counts()
    order = np.random.default_rng(seed).permutation(sizes.index)
    target = TEST_SIZE * len(df)
    test_areas, total = [], 0
    for a in order:
        if abs(total + sizes[a] - target) > abs(total - target):
            continue                 # this area overshoots; try the next
        test_areas.append(a)
        total += sizes[a]
    is_test = np.isin(areas, test_areas)
    return np.where(~is_test)[0], np.where(is_test)[0]


SPLITTERS = {
    "random": rand_split,
    "house": house_split,
    "temporal": temp_split,
    "spatial": spatial_split,
}


def make_split(df, scheme, seed):
    return SPLITTERS[scheme](df, seed)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    from data import load
    df = load()
    rows = []
    for scheme in SPLITTERS:
        tr, te = make_split(df, scheme, seed=42)
        train_ids = set(df["id"].iloc[tr])
        rows.append({
            "scheme": scheme,
            "train": len(tr),
            "test": len(te),
            "test %": round(100 * len(te) / len(df), 1),
            "overlap rows": len(set(tr) & set(te)),          # must be 0
            "test houses also in train": int(
                df["id"].iloc[te].isin(train_ids).sum()),
            "test dates": f"{df['date'].iloc[te].min().date()} to "
                          f"{df['date'].iloc[te].max().date()}",
        })
    print(pd.DataFrame(rows).to_string(index=False))
