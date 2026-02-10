"""Command-line interface."""

import click


@click.group()
@click.version_option(package_name="crows-nest")
def main() -> None:
    """Crow's Nest: A thunk-based multi-agent system."""


@main.command()
def hello() -> None:
    """Placeholder command for testing CLI setup."""
    click.echo("Crow's Nest is ready.")
