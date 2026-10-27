from tokyo_housing_lab import DISCLAIMER
from tokyo_housing_lab.cli import main


def test_disclaimer_text():
    assert DISCLAIMER.startswith("このサービスは、国土交通省の")
    assert DISCLAIMER.endswith("保証されたものではありません")


def test_cli_help_shows_disclaimer(capsys):
    main([])
    assert DISCLAIMER in capsys.readouterr().out
