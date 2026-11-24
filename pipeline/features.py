"""Stage 2 · Interpretable Feature Engineering (`pipeline/features.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: row_summary_stats, top_pair_ratios, cat_groupby_stats
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_FEATURE_OPS: tuple[str, ...] = ('row_summary_stats', 'top_pair_ratios', 'cat_groupby_stats')
PRIORITY_SIGNAL_COLUMNS: list[str] = ['conditioning_intensity', 'year_hct', 'dri_score', 'prim_disease_hct', 'comorbidity_score', 'gvhd_proph']


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit, human-readable engineered features one transformation per block."""
    out = df.copy()
    if not ACTIVE_FEATURE_OPS:
        return out

    fallback_cols = [c for c in PRIORITY_SIGNAL_COLUMNS if c in out.columns]
    if len(fallback_cols) < 2:
        for col in out.columns:
            if col not in fallback_cols and not str(col).endswith(("_isna", "_target_enc")):
                fallback_cols.append(str(col))
            if len(fallback_cols) >= 4:
                break

    # Safe domain ratios between top numeric predictors
    if 'conditioning_intensity' in out.columns and 'year_hct' in out.columns:
        _den = out['year_hct'].where(out['year_hct'].abs() >= 0.15, 0.15)
        out['conditioning_intensity_div_year_hct'] = (out['conditioning_intensity'] / _den).clip(-10.0, 10.0)
    if len(fallback_cols) >= 2:
        _a, _b = fallback_cols[0], fallback_cols[1]
        _rname = f'{_a}_div_{_b}'
        if _rname not in out.columns:
            _den = out[_b].where(out[_b].abs() >= 0.15, 0.15)
            out[_rname] = (out[_a] / _den).clip(-10.0, 10.0)
    # Row-wise summary moments across predictors
    if fallback_cols:
        out['row_numeric_mean'] = out[fallback_cols].mean(axis=1)
        out['row_numeric_std'] = out[fallback_cols].std(axis=1).fillna(0.0)
    if len(fallback_cols) >= 2:
        _rk = out[fallback_cols[0]].rank(method='first')
        _q = pd.qcut(_rk, q=4, labels=False, duplicates='drop')
        _g = out.groupby(_q)[fallback_cols[1]].transform('mean')
        out['cohort_ctx_mean'] = _g
        out['cohort_ctx_resid'] = out[fallback_cols[1]] - _g

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_features_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Alias preserving modular orchestrator compatibility."""
    return feature_engineering(df)
