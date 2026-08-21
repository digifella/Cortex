import pytest
from scripts.photo_archive import cli


def test_apply_defaults_to_false():
    args = cli.parse_args(["quarantine", "--db", "x.db", "--plan", "p.csv"])
    assert args.apply is False


def test_apply_flag_is_explicit():
    args = cli.parse_args(["quarantine", "--db", "x.db", "--plan", "p.csv",
                           "--apply"])
    assert args.apply is True


def test_all_stages_are_registered():
    for stage in ("walk", "exif", "hash", "plan-dupes", "quarantine",
                  "plan-organise", "organise", "undo"):
        assert cli.parse_args([stage, "--db", "x.db"]).command == stage


def test_unknown_stage_exits_nonzero():
    with pytest.raises(SystemExit):
        cli.parse_args(["banana", "--db", "x.db"])
