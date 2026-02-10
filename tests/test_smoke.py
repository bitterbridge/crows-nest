"""Smoke tests to verify basic package setup."""


def test_package_imports() -> None:
    """Verify that the main package can be imported."""
    import crows_nest

    assert hasattr(crows_nest, "__version__")
    assert crows_nest.__version__ == "0.1.0"


def test_submodules_import() -> None:
    """Verify that all submodules can be imported."""
    from crows_nest import agents, api, cli, core, mocks, observability, persistence, plugins

    # Just verify imports don't raise
    assert core is not None
    assert cli is not None
    assert api is not None
    assert agents is not None
    assert persistence is not None
    assert mocks is not None
    assert observability is not None
    assert plugins is not None


def test_cli_main_exists() -> None:
    """Verify that the CLI entry point exists."""
    from crows_nest.cli import main

    assert callable(main)
