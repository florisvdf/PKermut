import pytest
from pathlib import Path
import os
from pg_model.estimator import KermutEstimator


@pytest.fixture(scope="session")
def params(tmpdir_session):
    params = dict(
        data_artifact_path="/opt/ml/input/data/training",
        dataset_name="RASK_HUMAN_Weng_2022_binding-DARPin_K55",
        reference_sequence="MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTA"
                           "GQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKC"
                           "DLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKK"
                           "KKKSKTKCVIM",
        output_path=f"{tmpdir_session}/output/data",
        n_steps=2,
        device="cpu",
        preferential=False,
    )
    return params


@pytest.fixture(scope="session")
def estimator(params):
    estimator = KermutEstimator(
        role=os.environ.get("AWS_EXECUTION_ROLE"),
        hyperparameters=params,
        instance_type="local"
    )
    return estimator


def test_local_training_job(estimator, data_artifact_path):
    estimator.fit(
        inputs=str(Path(data_artifact_path).as_uri()),
        wait=False
    )
