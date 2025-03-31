from datetime import date, timedelta

import pandas as pd
import polars as pl
from implicit.als import AlternatingLeastSquares
from rectools.models import ImplicitALSWrapperModel

from recommenders.personal_top_popular import PersonalTopPopularV2, PersonalTopPopular
from recommenders.top_popular import TopPopular
from recommenders.als import ALS
from recommenders.two_level_with_ranker import TwoLevelRanker
from utils import prepare_test_data, calc_anp_print_metrics

TEST_START = date(2024, 7, 1)
FINAL_TOP_K = 30


def run_and_calc_metrics_top_pop():
    user_actions_full = pl.read_parquet('data/user_actions_full')

    sample_users = prepare_test_data(user_actions_full, TEST_START)

    train_orders = (
        user_actions_full
        .filter(pl.col('date') < TEST_START)
        .filter(pl.col('date') >= TEST_START - timedelta(days=3 * 30))
        .filter(pl.col('action_type') == 'order')
        .select('user_id', 'product_id')
    )
    top_popular = TopPopular()
    personal_top_popular = PersonalTopPopular()
    recommender = PersonalTopPopularV2(
        top_popular=top_popular,
        personal_top_popular=personal_top_popular,
    )
    recommender.fit(train_orders)
    recommend_sample_users = (
        sample_users
        .with_columns([
            pl.col('user_id').map_elements(
                lambda x: recommender.predict(user_id=x, top_k=FINAL_TOP_K),
                return_dtype=pl.List(pl.Int64)
            ).alias('recommendations'),
        ])
    )
    interactions = (
        recommend_sample_users
        .select('user_id', pl.col('ids').alias('item_id'))
        .explode('item_id')
    ).to_pandas()
    recommendations = (
        recommend_sample_users
        .select('user_id', pl.col('recommendations').alias('item_id'), pl.lit(1).alias('rank'))
        .explode('item_id')
    ).to_pandas()
    calc_anp_print_metrics(recommendations, interactions, top_k=FINAL_TOP_K)


def run_and_calc_metrics_als():
    user_actions_full = pl.read_parquet('data/user_actions_full')
    sample_users = prepare_test_data(user_actions_full, TEST_START)
    train_orders = (
        user_actions_full
        .filter(pl.col('date') < TEST_START)
        .filter(pl.col('date') >= TEST_START - timedelta(days=3 * 30))
        .filter(pl.col('action_type') == 'order')
        .select('user_id', 'product_id', 'date')
    )
    model = ImplicitALSWrapperModel(
        AlternatingLeastSquares(
            factors=64,
            regularization=0.01,
            alpha=1.0,
            random_state=0,
            use_gpu=False,
            num_threads=32,
            iterations=15,
        ),
    )
    als = ALS(
        rectools_model=model,
        model_path='./models/implicit_als_0402_0630.pkl',
    )
    als.fit(train_orders)
    interactions = (
        sample_users
        .select('user_id', pl.col('ids').alias('item_id'))
        .explode('item_id')
    ).to_pandas()
    recommendations = (
        als.predict_df(user_id=sample_users['user_id'].to_list(), top_k=FINAL_TOP_K, filter_viewed=False)
    )
    calc_anp_print_metrics(recommendations, interactions, top_k=FINAL_TOP_K)
    recommendations = (
        als.predict_df(user_id=sample_users['user_id'].to_list(), top_k=FINAL_TOP_K, filter_viewed=True)
    )
    calc_anp_print_metrics(recommendations, interactions, top_k=FINAL_TOP_K)


def run_and_calc_metrics_2_level():
    user_actions_full = (
        pl.read_parquet('data/user_actions_full')
    )
    sample_users = prepare_test_data(user_actions_full, TEST_START)

    data = (
        user_actions_full
        .filter(pl.col('date') < TEST_START)
        .filter(pl.col('date') >= TEST_START - timedelta(days=3 * 30))
    )
    model = ImplicitALSWrapperModel(
        AlternatingLeastSquares(
            factors=64,
            regularization=0.01,
            alpha=1.0,
            random_state=0,
            use_gpu=False,
            num_threads=32,
            iterations=15,
        ),
    )
    als = ALS(
        rectools_model=model,
        model_path='./models/implicit_als_0402_0630.pkl',
    )
    top_popular = TopPopular()
    personal_top_popular = PersonalTopPopular()
    ptp_v2 = PersonalTopPopularV2(
        top_popular=top_popular,
        personal_top_popular=personal_top_popular,
    )
    tlr = TwoLevelRanker(
        l1_info=[
            (ptp_v2, 100),
            (als, 100),
        ],
        ranker_path='./models/ranker_v1.bin',
    )
    tlr.fit(data)
    recommendations = pd.DataFrame({})
    for user_id in sample_users['user_id'].to_list():
        current_predict = tlr.predict(user_id, top_k=FINAL_TOP_K)
        recommendations = pd.concat([recommendations, current_predict])
    interactions = (
        sample_users
        .select('user_id', pl.col('ids').alias('item_id'))
        .explode('item_id')
    ).to_pandas()
    calc_anp_print_metrics(recommendations, interactions, top_k=FINAL_TOP_K)


if __name__ == '__main__':
    # run_and_calc_metrics_top_pop()
    # run_and_calc_metrics_als()
    run_and_calc_metrics_2_level()
