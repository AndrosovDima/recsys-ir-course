from datetime import date

import polars as pl
import pandas as pd
from rectools.metrics import (
    Precision,
    NDCG,
    Recall,
    MAP,
    calc_metrics,
)

FINAL_TOP_K = 30


def calc_anp_print_metrics(recommendations: pd.DataFrame, interactions: pd.DataFrame, top_k: int = FINAL_TOP_K):
    metrics = {
        "recall": Recall(k=top_k),
        "precision": Precision(k=top_k),
        "ndcg": NDCG(k=top_k),
        "map_": MAP(k=top_k),
    }
    metrics = calc_metrics(
        metrics=metrics,
        reco=recommendations,
        interactions=interactions,
    )
    print(metrics)


def prepare_test_data(user_actions_full: pl.DataFrame, test_start: date) -> pl.DataFrame:
    test_orders = (
        user_actions_full
        .filter(pl.col('date') >= test_start)
        .filter(pl.col('action_type') == 'order')
        .select('user_id', 'product_id')
    )
    sample_users = (
        test_orders
        .group_by('user_id')
        .agg(
            pl.col("product_id").unique().alias("ids")
        )
        .sort(by="user_id")
        .sample(n=1000, seed=0)
    )
    return sample_users
