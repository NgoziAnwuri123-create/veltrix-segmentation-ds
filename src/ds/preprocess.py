"""Imputation, transformation and scaling for the segmentation model.

Everything is learned in fit() and replayed in transform(), so training and scoring new customers
apply identical rules. Output: a DataFrame with scaled numeric columns, then categorical columns as strings.

Price features: the stated budget that applies to the customer's intent (sale budget for buyers, monthly
rent for renters) becomes log price, z-scored within intent -> `price_level`. The same is done for the median
price viewed -> `viewed_price_level`. Buyers and renters therefore share one comparable feature with no
structural NULLs. A missing stated budget is filled from the viewed price, then from the median of the customer's
(intent, customer type, region) group, then from the intent median; `budget_missing` keeps the fact visible.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.ds.feature_spec import MODEL_FEATURES

PRICE_GROUP = ["primary_intent", "customer_type", "region_of_residence"]


def _log_prices(df: pd.DataFrame) -> pd.DataFrame:
    rent = df.primary_intent.eq("Rent")
    stated = np.where(rent, df.rent_budget_pcm_gbp, df.budget_gbp).astype(float)
    viewed = np.where(rent, df.med_rent_viewed_pcm_gbp, df.med_asking_viewed_gbp).astype(float)
    return pd.DataFrame({"stated": np.log(stated), "viewed": np.log(viewed)}, index=df.index)


class SegmentationPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, winsor=(0.01, 0.99), corr_threshold=0.9):
        self.winsor = winsor
        self.corr_threshold = corr_threshold

    # -- helpers ---------------------------------------------------------------------------------
    def _categoricals(self, df):
        return df[self.cat_cols_].astype(object).where(df[self.cat_cols_].notna(), "Unknown").astype(str)

    def _price_levels(self, df):
        p = _log_prices(df)
        keys = df[PRICE_GROUP].astype(object).fillna("Unknown").apply(tuple, axis=1)
        stated = p.stated.fillna(p.viewed)
        stated = stated.fillna(keys.map(self.price_group_median_))
        stated = stated.fillna(df.primary_intent.map(self.price_intent_median_))
        viewed = p.viewed.fillna(stated)
        mu = df.primary_intent.map(self.price_intent_mean_)
        sd = df.primary_intent.map(self.price_intent_std_)
        return pd.DataFrame({"price_level": (stated - mu) / sd, "viewed_price_level": (viewed - mu) / sd}, index=df.index)

    def _numeric_raw(self, df):
        out = {}
        for f in self.num_specs_:
            x = df[f.name].astype(float)
            if f.kind == "flag":
                x = x.fillna(0.0)
            elif f.kind == "count" or f.log:
                x = np.log1p(x)
            out[f.name] = x
        return pd.DataFrame(out, index=df.index).join(self._price_levels(df))

    # -- sklearn API ----------------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y=None):
        self.cat_cols_ = [f.name for f in MODEL_FEATURES if f.kind == "cat"]
        self.num_specs_ = [f for f in MODEL_FEATURES if f.kind in ("flag", "count", "share", "num")]

        p = _log_prices(df)
        keys = df[PRICE_GROUP].astype(object).fillna("Unknown").apply(tuple, axis=1)
        self.price_group_median_ = p.stated.groupby(keys).median().dropna().to_dict()
        self.price_intent_median_ = p.stated.groupby(df.primary_intent).median().to_dict()
        filled = p.stated.fillna(p.viewed).fillna(keys.map(self.price_group_median_)).fillna(
            df.primary_intent.map(self.price_intent_median_))
        self.price_intent_mean_ = filled.groupby(df.primary_intent).mean().to_dict()
        self.price_intent_std_ = filled.groupby(df.primary_intent).std().to_dict()

        raw = self._numeric_raw(df)
        self.impute_ = raw.median()
        raw = raw.fillna(self.impute_)
        self.clip_low_ = raw.quantile(self.winsor[0])
        self.clip_high_ = raw.quantile(self.winsor[1])
        raw = raw.clip(self.clip_low_, self.clip_high_, axis=1)
        self.mean_, self.std_ = raw.mean(), raw.std()

        constant = self.std_[self.std_ < 1e-9].index.tolist()
        scaled = ((raw - self.mean_) / self.std_).drop(columns=constant)
        corr = scaled.corr().abs()
        kept, self.dropped_ = [], {c: "constant after winsorising" for c in constant}
        for col in scaled.columns:
            partner = next((k for k in kept if corr.loc[col, k] > self.corr_threshold), None)
            if partner is None:
                kept.append(col)
            else:
                self.dropped_[col] = f"|r|={corr.loc[col, partner]:.2f} with {partner}"
        self.num_cols_ = kept
        self.categories_ = {c: sorted(self._categoricals(df)[c].unique()) for c in self.cat_cols_}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = self._numeric_raw(df).fillna(self.impute_)
        raw = raw.clip(self.clip_low_, self.clip_high_, axis=1)
        scaled = ((raw - self.mean_) / self.std_)[self.num_cols_]
        return pd.concat([scaled, self._categoricals(df)], axis=1)

    @property
    def feature_names_out_(self) -> list[str]:
        return self.num_cols_ + self.cat_cols_
