import circadian_jepa


def test_version_exists() -> None:
    assert hasattr(circadian_jepa, "__version__")
    assert isinstance(circadian_jepa.__version__, str)
    assert circadian_jepa.__version__ != ""
