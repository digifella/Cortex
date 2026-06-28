import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts import photo_batch as pb


def test_description_is_bad_empty():
    assert pb.description_is_bad("") is True
    assert pb.description_is_bad(None) is True
    assert pb.description_is_bad("   ") is True


def test_description_is_bad_placeholder():
    assert pb.description_is_bad("[Image: description timed out]") is True


def test_description_is_bad_too_short():
    assert pb.description_is_bad("A dog.") is True            # < 40 chars
    assert pb.description_is_bad("A dog.", min_len=3) is False


def test_description_is_bad_refusal_prefix_even_when_long():
    text = "I must give a thorough and complete description of this scene before I continue"
    assert pb.description_is_bad(text) is True


def test_description_is_good():
    good = ("A wooden sailboat moored at a stone jetty under an overcast sky, "
            "with green hills rising behind the harbour.")
    assert pb.description_is_bad(good) is False
