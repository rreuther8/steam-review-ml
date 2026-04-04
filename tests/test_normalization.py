"""
Tests for steam_review_ml.transforms.normalization.

Run: python -m unittest tests.test_normalization
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from steam_review_ml.transforms.normalization import (
    RULE_CAP_QUANTILE_LOG1P,
    RULE_LOG1P,
    add_normalized_columns,
    fit_normalization,
    inverse_norm_votes_helpful,
    make_norm_col_name,
)


class TestMakeNormColName(unittest.TestCase):
    def test_dots_replaced(self):
        self.assertEqual(make_norm_col_name("author.playtime_at_review"), "_norm_author__playtime_at_review")

    def test_no_dots(self):
        self.assertEqual(make_norm_col_name("votes_helpful"), "_norm_votes_helpful")


class TestFitNormalization(unittest.TestCase):
    def test_skips_missing_columns(self):
        train = pd.DataFrame({"votes_helpful": [0, 1, 2, 100]})
        rules = {
            "votes_helpful": {"kind": RULE_CAP_QUANTILE_LOG1P, "quantile": 0.99},
            "author.playtime_at_review": {"kind": RULE_LOG1P},
        }
        fitted = fit_normalization(train, rules=rules)
        self.assertIn("votes_helpful", fitted)
        self.assertNotIn("author.playtime_at_review", fitted)

    def test_cap_is_train_quantile(self):
        train = pd.DataFrame({"votes_helpful": np.arange(0, 101, dtype=float)})
        rules = {"votes_helpful": {"kind": RULE_CAP_QUANTILE_LOG1P, "quantile": 0.99}}
        fitted = fit_normalization(train, rules=rules)
        self.assertEqual(fitted["votes_helpful"]["kind"], RULE_CAP_QUANTILE_LOG1P)
        self.assertAlmostEqual(float(fitted["votes_helpful"]["cap"]), 99.0, places=5)

    def test_unknown_kind_raises(self):
        train = pd.DataFrame({"x": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            fit_normalization(train, rules={"x": {"kind": "nope"}})


class TestAddNormalizedColumns(unittest.TestCase):
    def test_log1p_matches_numpy(self):
        fitted = {"review_word_count": {"kind": RULE_LOG1P}}
        df = pd.DataFrame({"review_word_count": [0.0, 3.0, 10.0]})
        out = add_normalized_columns(df, fitted)
        expected = np.log1p([0.0, 3.0, 10.0])
        np.testing.assert_array_almost_equal(
            out["_norm_review_word_count"].to_numpy(),
            expected,
        )
        self.assertTrue("review_word_count" in out)

    def test_cap_then_log1p_clamps_high_values(self):
        fitted = {"votes_helpful": {"kind": RULE_CAP_QUANTILE_LOG1P, "cap": 10.0}}
        df = pd.DataFrame({"votes_helpful": [0.0, 10.0, 500.0]})
        out = add_normalized_columns(df, fitted)
        # Both 10 and 500 cap to 10 before log1p
        np.testing.assert_array_almost_equal(
            out.loc[out["votes_helpful"].isin([10.0, 500.0]), "_norm_votes_helpful"].to_numpy(),
            [np.log1p(10.0), np.log1p(10.0)],
        )

    def test_inplace(self):
        fitted = {"votes_helpful": {"kind": RULE_LOG1P}}
        df = pd.DataFrame({"votes_helpful": [1.0]})
        out = add_normalized_columns(df, fitted, inplace=True)
        self.assertIs(out, df)
        self.assertIn("_norm_votes_helpful", df)


class TestInverseNormVotesHelpful(unittest.TestCase):
    def test_roundtrip_below_cap(self):
        # Cap above all values so forward transform does not clip (tiny train sets
        # can yield q99 < max, which breaks exact roundtrip for the largest row).
        fitted = {"votes_helpful": {"kind": RULE_CAP_QUANTILE_LOG1P, "cap": 100.0, "quantile": 0.99}}
        df = pd.DataFrame({"votes_helpful": [0.0, 1.0, 5.0, 20.0]})
        norm = add_normalized_columns(df, fitted)["_norm_votes_helpful"].to_numpy()
        raw = inverse_norm_votes_helpful(norm, fitted)
        np.testing.assert_array_almost_equal(raw, [0.0, 1.0, 5.0, 20.0])

    def test_inverse_respects_cap(self):
        fitted = {"votes_helpful": {"kind": RULE_CAP_QUANTILE_LOG1P, "cap": 10.0, "quantile": 0.99}}
        # Huge normalized value would inverse to huge raw without cap clip
        y_norm = np.array([np.log1p(10.0) + 5.0])
        raw = inverse_norm_votes_helpful(y_norm, fitted)
        self.assertAlmostEqual(float(raw[0]), 10.0)

    def test_log1p_only_votes_helpful(self):
        fitted = {"votes_helpful": {"kind": RULE_LOG1P}}
        y = np.array([0.0, 2.0])
        y_norm = np.log1p(y)
        raw = inverse_norm_votes_helpful(y_norm, fitted)
        np.testing.assert_array_almost_equal(raw, y)


if __name__ == "__main__":
    unittest.main()
