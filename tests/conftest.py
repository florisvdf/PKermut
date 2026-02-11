import pytest
import os
from tempfile import mkdtemp
from pathlib import Path
import shutil


@pytest.fixture(scope="session")
def tmpdir_session() -> Path:
    path = mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture(scope="session")
def data_artifact_path():
    return Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def tcrg1_mouse_pdb_file():
    return Path(__file__).resolve().parent / "data/structures/pdbs/TCRG1_MOUSE.pdb"


@pytest.fixture(scope="session")
def default_bucket():
    return os.environ.get("AWS_DEFAULT_BUCKET")
