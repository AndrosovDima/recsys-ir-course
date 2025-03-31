from typing import List

import polars as pl

from recommenders.constants import DEFAULT_USER, DEFAULT_TOP_K
from recommenders.base_recommender import BaseRecommender
from recommenders.top_popular import TopPopular


class PersonalTopPopular(BaseRecommender):
    def __init__(self):
        self.sort_data = None
        self.id_name = None
        self.user_id = None

    def fit(self, data: pl.DataFrame, id_name: str = 'product_id', user_id: str = 'user_id'):
        self.sort_data = (
            data
            .group_by(user_id, id_name)
            .len()
            .sort('len', descending=True)
        )
        self.id_name = id_name
        self.user_id = user_id

    def predict(self, user_id: int = DEFAULT_USER, top_k: int = DEFAULT_TOP_K) -> List[int]:
        return (
            self.sort_data
            .filter(pl.col(self.user_id) == user_id)
            .limit(top_k)[self.id_name]
        ).to_list()


class PersonalTopPopularV2(BaseRecommender):
    def __init__(self, top_popular: TopPopular, personal_top_popular: PersonalTopPopular):
        self.top_popular = top_popular
        self.personal_top_popular = personal_top_popular

    def fit(self, data: pl.DataFrame, id_name: str = 'product_id', user_id: str = 'user_id'):
        self.top_popular.fit(data, id_name)
        self.personal_top_popular.fit(data, id_name, user_id)

    def predict(self, user_id: int = DEFAULT_USER, top_k: int = DEFAULT_TOP_K) -> List[int]:
        recs = []
        pers_top = self.personal_top_popular.predict(user_id, top_k)
        if len(pers_top) < top_k:
            top = self.top_popular.predict(user_id, top_k - len(pers_top))
            recs = pers_top + top
        else:
            recs = pers_top
        return recs
