"""Stage 2 · Interpretable Feature Engineering (`pipeline/features.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: synth:groupby_z:comorbidity_score:dri_score, synth:cat_cross:dri_score:gvhd_proph
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_FEATURE_OPS: tuple[str, ...] = ('synth:groupby_z:comorbidity_score:dri_score', 'synth:cat_cross:dri_score:gvhd_proph')
PRIORITY_SIGNAL_COLUMNS: list[str] = ['year_hct', 'conditioning_intensity', 'dri_score', 'prim_disease_hct', 'cyto_score_detail', 'age_at_hct']


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

    if 'comorbidity_score' in out.columns and 'dri_score' in out.columns:
        _grp = out['dri_score'].round(3)
        _gmu = out.groupby(_grp)['comorbidity_score'].transform('mean')
        _gsd = out.groupby(_grp)['comorbidity_score'].transform('std').fillna(1.0)
        out['comorbidity_score_groupby_z_dri_score'] = ((out['comorbidity_score'] - _gmu) / _gsd.clip(lower=1e-2)).clip(-10, 10)
    if 'dri_score' in out.columns and 'gvhd_proph' in out.columns:
        _pa = out['dri_score'].round(3).astype(str)
        _pair = _pa + '__' + out['gvhd_proph'].round(3).astype(str)
        _vc = _pair.value_counts(normalize=True)
        out['dri_score_x_gvhd_proph_freq'] = np.log1p(100.0 * _pair.map(_vc).fillna(0.0))

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_features_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Alias preserving modular orchestrator compatibility."""
    return feature_engineering(df)
