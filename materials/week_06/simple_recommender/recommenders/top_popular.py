from typing import List

import polars as pl

from recommenders.constants import DEFAULT_USER, DEFAULT_TOP_K
from recommenders.base_recommender import BaseRecommender


class TopPopular(BaseRecommender):
    def __init__(self):
        self.sort_data = None
        self.id_name = None

    def fit(self, data: pl.DataFrame, id_name: str = 'product_id', user_id: str = 'user_id'):
        self.sort_data = (
            data
            .group_by(id_name)
            .len()
            .sort('len', descending=True)
        )
        self.id_name = id_name

    def predict(self, _: int = DEFAULT_USER, top_k: int = DEFAULT_TOP_K) -> List[int]:
        return self.sort_data.limit(top_k)[self.id_name].to_list()
