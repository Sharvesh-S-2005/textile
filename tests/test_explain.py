"""Tests for ml/explain.py helpers."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml.explain import explainer_inputs, family, readable
from ml.forecast import make_random_forest


def test_readable_labels() -> None:
    assert readable("noise_dba_lag5") == "noise, 5 min ago"
    assert readable("pm25_ugm3_rmean15") == "PM2.5, 15-min mean"
    assert readable("max_vib_rms_g") == "max machine vibration"
    assert readable("noise_dba") == "noise now"


def test_feature_families() -> None:
    assert family("noise_dba_rstd30") == family("wbgt_c_lag1")
    assert family("is_break") == family("minute_sin")
    assert family("mean_current_a") == "machine state"


def test_explainer_inputs_apply_the_pipeline_imputer() -> None:
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    pipe = make_random_forest({"n_estimators": 5}, 0).fit(X, [1, 2, 3, 4])
    tree, Xt = explainer_inputs(pipe, X)
    assert isinstance(tree, RandomForestRegressor) and not np.isnan(Xt).any()
