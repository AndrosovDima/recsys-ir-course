import typing as tp

import polars as pl
import pandas as pd

from recommenders.constants import DEFAULT_USER, DEFAULT_TOP_K


class BaseRecommender:
    def fit(self, data: pl.DataFrame, id_name: str = 'product_id', user_id: str = 'user_id'):
        # abstract method
        raise NotImplementedError

    def predict(
            self, user_id: tp.Union[int, tp.List[int]] = DEFAULT_USER, top_k: int = DEFAULT_TOP_K
    ) -> tp.Union[tp.List[int], pd.DataFrame, pl.DataFrame]:
        # abstract method
        raise NotImplementedError
