"""
Tests for filtering and feature selection in steam_review_ml.data.preprocess.

Validates row-level filters (language, empty review, negative playtime, vote
sentinel) and that select_features produces the expected columns and derived
fields. Uses small in-memory DataFrames; no raw CSV required.

Run: python -m unittest tests.test_preprocess
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from steam_review_ml.data.preprocess import (
    add_review_age_seconds,
    feature_engineering,
    filter_reviews,
    select_features,
    train_max_timestamp_created,
)


def _minimal_raw_df(overrides=None):
    """Minimal DataFrame with columns expected by filter_reviews / select_features."""
    raw = pd.DataFrame(
        {
            "language": ["english", "english", "spanish", "english"],
            "review": ["Good game.", "Okay.", "Buen juego.", "x"],  # last is short
            "recommended": [True, False, True, True],
            "votes_helpful": [0, 1, 0, 0],
            "votes_funny": [0, 0, 0, 0],
            "comment_count": [0, 0, 0, 0],
            "app_id": [1, 1, 2, 1],
            "app_name": ["A", "A", "B", "A"],
            "review_id": [10, 11, 12, 13],
            "author.steamid": ["u1", "u2", "u3", "u4"],
            "author.num_games_owned": [5, 10, 3, 0],
            "author.num_reviews": [1, 2, 0, 0],
            "author.playtime_last_two_weeks": [0.0, 10.0, 5.0, 0.0],
            "author.playtime_at_review": [100.0, 200.0, 50.0, 0.0],
            "steam_purchase": [True, True, False, True],
            "received_for_free": [False, False, True, False],
            "written_during_early_access": [False, False, False, False],
            "timestamp_created": [1000000, 1000001, 1000002, 1000003],
            "timestamp_updated": [1000000, 1000001, 1000002, 1000003],
            "author.last_played": [999000, 999001, 999002, 999003],
        }
    )
    if overrides:
        for k, v in overrides.items():
            raw[k] = v
    return raw


class TestFilterReviews(unittest.TestCase):
    """Option B: one test per filter via filter_reviews(), one big pipeline test."""

    def test_drop_vote_sentinel(self):
        # Data that would pass language + review + playtime; only sentinel removes rows
        df = _minimal_raw_df()
        df = df[
            (df["language"] == "english") & (df["review"].str.strip().str.len() >= 4)
        ].copy()
        df.loc[df.index[0], "votes_helpful"] = 4294967295
        df.loc[df.index[1], "votes_funny"] = 4294967295
        out = filter_reviews(df)
        self.assertEqual(len(out), 0)
        self.assertTrue((out["votes_helpful"] < 4294967295).all() if len(out) else True)
        self.assertTrue((out["votes_funny"] < 4294967295).all() if len(out) else True)

    def test_filter_by_language(self):
        df = _minimal_raw_df()
        out = filter_reviews(df, language="english")
        self.assertTrue(out["language"].eq("english").all())
        out_es = filter_reviews(df, language="spanish")
        self.assertTrue(out_es["language"].eq("spanish").all())

    def test_drop_empty_or_short_review(self):
        df = _minimal_raw_df()
        out = filter_reviews(df)
        self.assertTrue(out["review"].str.strip().str.len().ge(4).all())

    def test_drop_negative_playtime(self):
        df = _minimal_raw_df()
        df.loc[0, "author.playtime_at_review"] = -1.0
        out = filter_reviews(df)
        self.assertTrue((out["author.playtime_at_review"] >= 0).all())

    def test_drop_missing_author_last_played(self):
        df = _minimal_raw_df()
        df.loc[0, "author.last_played"] = pd.NA
        out = filter_reviews(df)
        self.assertTrue(out["author.last_played"].notna().all())

    def test_full_pipeline_filter_then_select(self):
        """One big test: clean pipeline through select_features (FE after split in prod)."""
        df = _minimal_raw_df()
        n_before = len(df)
        filtered = filter_reviews(df)
        featured = select_features(filtered)
        self.assertLessEqual(len(featured), n_before)
        self.assertTrue((featured["votes_helpful"] < 4294967295).all())
        self.assertIn("is_helpful", featured.columns)
        self.assertNotIn("review_length_chars", featured.columns)
        self.assertNotIn("review_word_count", featured.columns)
        engineered = feature_engineering(featured)
        self.assertIn("review_length_chars", engineered.columns)
        self.assertIn("review_word_count", engineered.columns)


class TestSelectFeatures(unittest.TestCase):
    def test_adds_is_helpful_not_text_derived_counts(self):
        df = _minimal_raw_df()
        df = filter_reviews(df)
        out = select_features(df)
        self.assertIn("language", out.columns)
        self.assertIn("is_helpful", out.columns)
        self.assertTrue((out["is_helpful"] == (out["votes_helpful"] >= 1)).all())
        self.assertNotIn("review_length_chars", out.columns)
        self.assertNotIn("review_word_count", out.columns)
        fe = feature_engineering(out)
        self.assertIn("review_length_chars", fe.columns)
        self.assertTrue(fe["review_length_chars"].ge(0).all())

    def test_fills_playtime_nans_with_zero(self):
        df = _minimal_raw_df()
        df.loc[0, "author.playtime_last_two_weeks"] = pd.NA
        df = filter_reviews(df)
        out = select_features(df)
        self.assertTrue(out["author.playtime_last_two_weeks"].notna().all())

    def test_drops_unnamed_0_if_present(self):
        df = _minimal_raw_df()
        df["Unnamed: 0"] = range(len(df))
        df = filter_reviews(df)
        out = select_features(df)
        self.assertNotIn("Unnamed: 0", out.columns)


class TestReviewAge(unittest.TestCase):
    def test_train_max_timestamp_created(self):
        chunks = [
            pd.DataFrame({"timestamp_created": [100, 200]}),
            pd.DataFrame({"timestamp_created": [50, 300]}),
        ]
        self.assertEqual(train_max_timestamp_created(iter(chunks)), 300.0)

    def test_train_max_timestamp_created_empty_raises(self):
        with self.assertRaises(ValueError):
            train_max_timestamp_created(iter([pd.DataFrame({"timestamp_created": []})]))

    def test_add_review_age_seconds(self):
        df = pd.DataFrame({"timestamp_created": [1000, 1500, np.nan]})
        out = add_review_age_seconds(df, reference_timestamp=2000)
        self.assertIn("review_age_seconds", out.columns)
        self.assertAlmostEqual(float(out.loc[0, "review_age_seconds"]), 1000.0)
        self.assertAlmostEqual(float(out.loc[1, "review_age_seconds"]), 500.0)
        self.assertTrue(pd.isna(out.loc[2, "review_age_seconds"]))

    def test_add_review_age_seconds_clips_negative_to_zero(self):
        df = pd.DataFrame({"timestamp_created": [2500]})
        out = add_review_age_seconds(df, reference_timestamp=2000)
        self.assertAlmostEqual(float(out.loc[0, "review_age_seconds"]), 0.0)


if __name__ == "__main__":
    unittest.main()
