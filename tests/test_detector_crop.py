from __future__ import annotations

import pytest

from src.infer.detector_crop import (
    PersonDetection,
    build_union_crop_from_detections,
    clamp_bbox,
    expand_bbox,
    select_primary_detection,
    union_bboxes,
)


def test_expand_bbox_pads_aspect_and_clamps():
    got = expand_bbox(
        (40, 20, 60, 80),
        pad_ratio=0.25,
        width=100,
        height=100,
        aspect_ratio=0.5,
    )
    x1, y1, x2, y2 = got
    assert got == clamp_bbox(got, 100, 100)
    assert pytest.approx((x2 - x1) / (y2 - y1), rel=1e-3) == 0.5
    assert x1 < 40 and x2 > 60 and y1 < 20 and y2 > 80


def test_union_bboxes():
    assert union_bboxes([(10, 20, 30, 40), (5, 25, 50, 60)]) == (5.0, 20.0, 50.0, 60.0)


def test_select_primary_detection_tracks_previous_subject():
    previous = (10, 10, 60, 100)
    far_high_score = PersonDetection((120, 10, 190, 100), 0.99)
    overlapping_lower_score = PersonDetection((12, 12, 62, 102), 0.60)
    got = select_primary_detection(
        [far_high_score, overlapping_lower_score],
        previous_bbox=previous,
    )
    assert got == overlapping_lower_score


def test_build_union_crop_from_detections_records_metrics():
    result = build_union_crop_from_detections(
        [
            (0, [PersonDetection((20, 20, 70, 140), 0.9)]),
            (10, []),
            (20, [PersonDetection((25, 18, 75, 145), 0.8)]),
        ],
        frame_width=120,
        frame_height=160,
        aspect_ratio=0.75,
        pad_ratio=0.2,
        min_detection_rate=0.5,
        max_edge_contact_rate=1.0,
    )
    assert result.detected_count == 2
    assert result.sample_count == 3
    assert result.detection_rate == pytest.approx(2 / 3)
    assert result.raw_union_bbox_xyxy == (20.0, 18.0, 75.0, 145.0)
    assert result.accepted is True
    assert result.bbox_xyxy[0] <= 20 and result.bbox_xyxy[2] >= 75


def test_build_union_crop_falls_back_when_detector_misses():
    result = build_union_crop_from_detections(
        [(0, []), (10, [])],
        frame_width=120,
        frame_height=160,
        aspect_ratio=90 / 130,
        fallback_bbox=(10, 20, 100, 150),
    )
    assert result.detected_count == 0
    assert result.accepted is False
    assert result.fallback_reason == "no_detector_person_boxes"
    assert result.bbox_xyxy == (10.0, 20.0, 100.0, 150.0)
