## Reasoning Trace & Transfer Verification (Diverge → Verify → Converge)

- **Architecture**: `Two-Stage KM-Weighted Incidence + Counterfactual Log-Time + Multi-GBDT + TabM PLE` | **Outcome**: `C_INDEX` `0.608626` → **`0.687009`**
- **Sensor / Schema Profile**: regime=`censored_survival`, ref_curve=`False`, sensor_channels=`0`, sensor_missing_rate=`0.00`, disjoint_entities=`False`

### Divergent Options Evaluated Before Convergence
- **[REJECTED] `opt_naive_regression_on_time`** (`tabular_baseline`, metric=`0.60863`): Treats censored survivors identically to early deaths; severe concordance collapse.
- **[ADAPTED] `opt_single_km_target_gbdt`** (`survival_gbdt`, metric=`0.68213`): Strong diverse base learners (+0.0735 C_INDEX), retained inside multi-stream ensemble alongside Two-Stage Counterfactual.
- **[CONVERGED] `opt_two_stage_counterfactual_multistream`** (`two_stage_counterfactual`, metric=`0.68701`): Outperformed every solo survival branch with +0.00761 gain over prior 0.67940 commit and 0.0 zero-pickle parity.

### Machine-Verifiable Transfer Assumptions (For Weaker Models)
- **`require_censored_event_indicator`** (`hard_blocker`, `has_censored_event` `bool_true` `True`): Two-Stage incidence + counterfactual survival decomposition specifically models right-censored event indicators. → *If violated*: DIVERGE_DO_NOT_COPY survival decomposition; for uncensored regression/classification, use direct GBDT + TabM/RealMLP/Ridge.
- **`shared_demographic_strata_not_disjoint_entities`** (`adapt_required`, `disjoint_entity_groups` `bool_false` `False`): StratifiedKFold + OOF target encoding assume equity/demographic strata overlap between train and test. → *If violated*: ADAPT: if groups are disjoint entities (GroupKFold), exclude group IDs from features and use entity-grouped inner validation.

### Copy vs. Adapt vs. Diverge Rules
- DO NOT copy Two-Stage survival decomposition onto uncensored datasets lacking an event indicator.
- ALWAYS verify whether group columns are overlapping strata (keep as features) or disjoint entity IDs (exclude under GroupKFold).
