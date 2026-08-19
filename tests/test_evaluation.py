from pathlib import Path

import pytest

from evaluation.evaluate import (
    Row,
    intersection_over_union,
    match_boxes,
    parse_mot,
    suppress_ignored_predictions,
)


def row(identity: int, box: tuple[float, float, float, float]) -> Row:
    return Row(1, identity, box, 1.0)


def test_matching_is_one_to_one_at_threshold() -> None:
    targets = [row(1, (0, 0, 10, 10)), row(2, (20, 0, 10, 10))]
    predictions = [
        row(-1, (0, 0, 10, 10)),
        row(-1, (1, 0, 10, 10)),
        row(-1, (20, 0, 10, 10)),
    ]

    matches = match_boxes(targets, predictions, 0.5)

    assert len(matches) == 2
    assert len({prediction for _, prediction in matches}) == 2


def test_ignore_suppression_protects_target_match() -> None:
    target = row(1, (0, 0, 10, 10))
    ignored = row(2, (0, 0, 10, 10))
    prediction = row(10, (0, 0, 10, 10))

    assert suppress_ignored_predictions([target], [ignored], [prediction], 0.5) == [
        prediction
    ]
    assert suppress_ignored_predictions([], [ignored], [prediction], 0.5) == []


def test_parse_mot_ground_truth(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text("1,42,10,20,30,40,1,1,0.75\n")

    parsed = parse_mot(gt_path, ground_truth=True)

    assert parsed[1] == [Row(1, 42, (10, 20, 30, 40), 1, 1, 0.75)]


def test_parse_mot_rejects_invalid_box(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text("1,1,0,0,0,10,1,-1,-1,-1\n")

    with pytest.raises(ValueError, match="invalid frame or box"):
        parse_mot(path)


def test_intersection_over_union() -> None:
    assert intersection_over_union((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(
        1 / 3
    )
