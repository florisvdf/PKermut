import pytest
from pg_model.__main__ import train


def test_train(data_artifact_path, tmpdir_session):
    train(
        data_artifact_path=str(data_artifact_path),
        target="DMS_score",
        dataset_name="RASK_HUMAN_Weng_2022_binding-DARPin_K55",
        reference_sequence="MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTA"
                           "GQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKC"
                           "DLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKK"
                           "KKKSKTKCVIM",
        output_path=str(tmpdir_session),
        n_steps=2,
        device="cpu",
        preferential=False,
    )


def test_preferential_train(data_artifact_path, tmpdir_session):
    train(
        data_artifact_path=str(data_artifact_path),
        target="DMS_score",
        dataset_name="RASK_HUMAN_Weng_2022_binding-DARPin_K55",
        reference_sequence="MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTA"
                           "GQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKC"
                           "DLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKK"
                           "KKKSKTKCVIM",
        output_path=str(tmpdir_session),
        preferential=True,
        preference_sampling_strategy="uniform_0.1",
        n_steps=2,
        device="cpu",
    )


def test_s3_train(data_artifact_path, tmpdir_session, default_bucket):
    train(
        data_artifact_path=f"s3://{default_bucket}/data/artifacts/test/data/",
        target="DMS_score",
        dataset_name="RASK_HUMAN_Weng_2022_binding-DARPin_K55",
        reference_sequence="MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTA"
                           "GQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKC"
                           "DLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKK"
                           "KKKSKTKCVIM",
        output_path=str(tmpdir_session),
        n_steps=2,
        device="cpu",
        dataset_uri=f"s3://{default_bucket}/data/test/RASK_HUMAN_Weng_2022_binding-DARPin_K55.csv"
    )


@pytest.mark.slow
def test_compute_artifacts_train(data_artifact_path, tmpdir_session, tcrg1_mouse_pdb_file):
    train(
        data_artifact_path=str(data_artifact_path),
        target="DMS_score",
        dataset_name="TCRG1_MOUSE_Tsuboyama_2023_1E0L",
        reference_sequence="GATAVSEWTEYKTADGKTYYYNNRTLESTWEKPQELK",
        pdb_path=str(tcrg1_mouse_pdb_file),
        output_path=str(tmpdir_session),
        n_steps=2,
        device="cpu",
        preferential=False,
    )
