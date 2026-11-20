"""Stage 2 · Interpretable Feature Engineering (`pipeline/features.py`).

Synthesized by sovLoop (Model: qwen36-architect)
Active Operations: plr_periodic_embeddings, pairwise_diff_crosses, cat_groupby_stats, log1p_skewed, top_pair_ratios
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_FEATURE_OPS: tuple[str, ...] = ('plr_periodic_embeddings', 'pairwise_diff_crosses', 'cat_groupby_stats', 'log1p_skewed', 'top_pair_ratios')
PRIORITY_SIGNAL_COLUMNS: list[str] = ['conditioning_intensity', 'dri_score', 'prim_disease_hct', 'gvhd_proph', 'year_hct', 'sex_match']


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

    # Signed log1p compression for skewed continuous predictors
    if 'conditioning_intensity' in out.columns: out['conditioning_intensity_log1p'] = np.sign(out['conditioning_intensity']) * np.log1p(np.abs(out['conditioning_intensity']))
    if 'dri_score' in out.columns: out['dri_score_log1p'] = np.sign(out['dri_score']) * np.log1p(np.abs(out['dri_score']))
    if 'prim_disease_hct' in out.columns: out['prim_disease_hct_log1p'] = np.sign(out['prim_disease_hct']) * np.log1p(np.abs(out['prim_disease_hct']))
    for _c in fallback_cols[:3]:
        if f'{_c}_log1p' not in out.columns:
            out[f'{_c}_log1p'] = np.sign(out[_c]) * np.log1p(np.abs(out[_c]))
    # Safe domain ratios between top numeric predictors
    if 'conditioning_intensity' in out.columns and 'dri_score' in out.columns:
        _den = out['dri_score'].where(out['dri_score'].abs() >= 0.15, 0.15)
        out['conditioning_intensity_div_dri_score'] = (out['conditioning_intensity'] / _den).clip(-10.0, 10.0)
    if len(fallback_cols) >= 2:
        _a, _b = fallback_cols[0], fallback_cols[1]
        _rname = f'{_a}_div_{_b}'
        if _rname not in out.columns:
            _den = out[_b].where(out[_b].abs() >= 0.15, 0.15)
            out[_rname] = (out[_a] / _den).clip(-10.0, 10.0)
    # Pairwise differences & absolute discrepancies
    if len(fallback_cols) >= 2:
        _a, _b = fallback_cols[0], fallback_cols[1]
        out[f'{_a}_minus_{_b}'] = out[_a] - out[_b]
        out[f'{_a}_absdiff_{_b}'] = (out[_a] - out[_b]).abs()
    # Smooth periodic numerical embeddings (sin/cos harmonics)
    for _c in fallback_cols[:3]:
        _scaled = out[_c].clip(-10.0, 10.0) * (2.0 * np.pi * 0.20)
        out[f'{_c}_sin'] = np.sin(_scaled)
        out[f'{_c}_cos'] = np.cos(_scaled)
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
