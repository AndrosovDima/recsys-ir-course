import typing as tp
import os

import polars as pl
import pandas as pd
from rectools.models import ImplicitALSWrapperModel, load_model
from rectools.dataset import Dataset

from recommenders.constants import DEFAULT_USER, DEFAULT_TOP_K
from recommenders.base_recommender import BaseRecommender


class ALS(BaseRecommender):
    def __init__(self, rectools_model: ImplicitALSWrapperModel, model_path: str):
        self.rectools_model = rectools_model
        self.model_path = model_path
        self.dataset: tp.Optional[Dataset] = None

    def fit(self, data: pl.DataFrame, id_name: str = 'product_id', user_id: str = 'user_id'):
        self.dataset = Dataset.construct(
            interactions_df=(
                data
                .select(
                    'user_id',
                    pl.col('product_id').alias('item_id'),
                    pl.lit(1).alias('weight'),
                    pl.col('date').alias('datetime')
                )
                .to_pandas()
            ),
        )
        if os.path.isfile(self.model_path):
            print("model exist")
            self.rectools_model = load_model(self.model_path)
        else:
            print("model not exist")
            self.rectools_model.fit(self.dataset)
            self.rectools_model.save(self.model_path)

    def predict(self, user_id: int = DEFAULT_USER, top_k: int = DEFAULT_TOP_K) -> tp.List[int]:
        recoms = self.rectools_model.recommend(
            users=[user_id],
            dataset=self.dataset,
            k=top_k,
            filter_viewed=False,
            on_unsupported_targets="ignore",
        )
        return recoms['item_id'].to_list()

    def predict_df(
            self, user_id: tp.List[int], top_k: int = DEFAULT_TOP_K, filter_viewed: bool = True
    ) -> pd.DataFrame:
        recoms = self.rectools_model.recommend(
            users=user_id,
            dataset=self.dataset,
            k=top_k,
            filter_viewed=filter_viewed,
            on_unsupported_targets="ignore",
        )
        return recoms
