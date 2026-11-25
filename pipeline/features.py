"""Stage 2 · Interpretable Feature Engineering (`pipeline/features.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: synth:groupby_z:hla_match_c_high:dri_score
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_FEATURE_OPS: tuple[str, ...] = ('synth:groupby_z:hla_match_c_high:dri_score',)
PRIORITY_SIGNAL_COLUMNS: list[str] = ['year_hct', 'conditioning_intensity', 'dri_score', 'prim_disease_hct', 'comorbidity_score', 'age_at_hct']


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

    if 'hla_match_c_high' in out.columns and 'dri_score' in out.columns:
        _grp = out['dri_score'].round(3)
        _gmu = out.groupby(_grp)['hla_match_c_high'].transform('mean')
        _gsd = out.groupby(_grp)['hla_match_c_high'].transform('std').fillna(1.0)
        out['hla_match_c_high_groupby_z_dri_score'] = ((out['hla_match_c_high'] - _gmu) / _gsd.clip(lower=1e-2)).clip(-10, 10)

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_features_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Alias preserving modular orchestrator compatibility."""
    return feature_engineering(df)
