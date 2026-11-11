import ai_learning_lab
from ai_learning_lab.utils import package_version, runtime_metadata


def test_package_version_missing():
    assert package_version("definitely-not-a-real-distribution-xyz") == "not-installed"


def test_package_version_installed():
    assert package_version("scikit-learn") != "not-installed"


def test_runtime_metadata_shape():
    metadata = runtime_metadata()
    assert set(metadata) == {"hardware", "dependencies"}
    assert metadata["hardware"]["python_version"].count(".") == 2


def test_package_exposes_version():
    assert ai_learning_lab.__version__ == "0.1.0"
