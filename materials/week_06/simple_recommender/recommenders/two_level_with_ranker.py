import os
import typing as tp

import polars as pl
import pandas as pd
import catboost
import numpy as np

from recommenders.constants import DEFAULT_USER, DEFAULT_TOP_K
from recommenders.base_recommender import BaseRecommender


class TwoLevelRanker(BaseRecommender):
    def __init__(self, l1_info: tp.List[tp.Tuple[BaseRecommender, int]], ranker_path: str):
        # predict
        # list candgen (als, topV2) + const num can
        # features из l1 -> ranker
        # ranker
        # sort score -> top_k

        # fit
        # ranker load
        # fit candgen
        # precalc feat
        self.l1_info = l1_info
        self.ranker_path = ranker_path
        self.ranker = None
        self.features_df = {}

    def fit(self, data: pl.DataFrame, id_name: str = 'product_id', user_id: str = 'user_id'):
        order_data = (
            data
            .filter(pl.col('action_type') == 'order')
        )
        for i in range(len(self.l1_info)):
            self.l1_info[i][0].fit(order_data, id_name, user_id)

        print('l1 fit')

        if os.path.isfile(self.ranker_path):
            print("ranker exist")
            self.ranker = catboost.CatBoost()
            self.ranker.load_model(self.ranker_path)
        else:
            print("model not exist")
            raise FileNotFoundError

        print('ranker load')
        # features
        for suf in ['click', 'favorite', 'to_cart', 'order']:
            self.features_df[f'{suf}_ui_features'] = (
                data
                .filter(pl.col('action_type') == suf)
                .group_by('user_id', 'product_id')
                .agg(
                    pl.count('product_id').alias(f'ui_num_{suf}')
                )
            )
            self.features_df[f'{suf}_i_features'] = (
                data
                .filter(pl.col('action_type') == suf)
                .group_by('product_id')
                .agg(
                    pl.count('user_id').alias(f'i_num_{suf}')
                )
            )
        print('features calc')

    def predict(
            self, user_id: tp.Union[int, tp.List[int]] = DEFAULT_USER, top_k: int = DEFAULT_TOP_K
    ) -> tp.Union[tp.List[int], pd.DataFrame, pl.DataFrame]:
        # predict
        # list candgen (als, topV2) + const num can
        # features из l1 -> ranker
        # ranker
        # sort score -> top_k
        candidates = []
        for i in range(len(self.l1_info)):
            candidates.extend(
                self.l1_info[i][0].predict(user_id, top_k=self.l1_info[i][1])
            )

        candidates_df = pl.DataFrame(
            data={"user_id": [user_id] * len(candidates), "product_id": candidates},
            schema={"user_id": pl.Int32, "product_id": pl.Int32}
        )

        for key, df in self.features_df.items():
            if 'ui' in key:
                candidates_df = (
                    candidates_df
                    .join(df, on=['user_id', 'product_id'], how='left')
                )
            else:
                candidates_df = (
                    candidates_df
                    .join(df, on=['product_id'], how='left')
                )

        candidates_df_pd = candidates_df.to_pandas()
        candidates_df_pd['predict'] = self.ranker.predict(candidates_df_pd, prediction_type='Probability')[:, 1]
        candidates_df_pd = (
            candidates_df_pd
            .sort_values(by=['predict'], ascending=False)
            .head(top_k)
        )
        candidates_df_pd['rank'] = np.arange(1, candidates_df_pd.shape[0] + 1)
        candidates_df_pd['item_id'] = candidates_df_pd['product_id']
        return candidates_df_pd[['user_id', 'item_id', 'rank']]
