"""Regression checks for benchmark splits, category IDs, and scorer inputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/baselines"))
from small_object_benchmark import prepare_view, predict
from small_object_data import Split, check_disjoint, coco_split, itsdt_development, svanstrom_splits, visdrone_split
from small_object_eval import score_coco, visdrone_predictions


def make_clip(root, name, category, category_id=2):
    images = []
    folder = root / "images" / name
    folder.mkdir(parents=True)
    for i in (10, 1, 2):
        filename = f"{name}/{i:07d}.jpg"
        Image.new("RGB", (64, 64), (60, 100, 140)).save(root / "images" / filename)
        images.append({"id": i, "video_id": 0, "frame_id": i, "file_name": filename, "width": 64, "height": 64})
    data = {"images": images, "categories": [{"id": category_id, "name": category}, {"id": 4, "name": "ignore"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": category_id,
                             "bbox": [12, 12, 8, 8], "area": 64, "iscrowd": 0},
                            {"id": 2, "image_id": 1, "category_id": 4,
                             "bbox": [25, 25, 20, 20], "area": 400, "iscrowd": 1}]}
    annotation = root / f"{name}.json"
    annotation.write_text(json.dumps(data))
    return coco_split(root, annotation, video=True)


def test_svanstrom_holds_out_later_clips_and_skips_boundaries(tmp_path):
    clips, videos, images = {}, [], []
    for category in ("DRONE", "BIRD"):
        for number in range(1, 11):
            clip = f"images/IR_{category}_{number:03}"
            image_id = len(images) + 1
            row = {"id": image_id, "video_id": image_id, "_relative_path": f"{clip}/1.jpg"}
            images.append(row)
            clips[clip] = [row]
            videos.append({"id": image_id, "source_class": category})
    release = Split(tmp_path, {"videos": videos, "images": images, "annotations": []}, clips)
    train, val, test = svanstrom_splits(release)
    check_disjoint(train, val, test, video=True)
    assert {key[-3:] for key in train.clips} == {"001", "002", "003", "004"}
    assert {key[-3:] for key in val.clips} == {"006", "007"}
    assert {key[-3:] for key in test.clips} == {"009", "010"}


def test_itsdt_uses_the_published_training_frames(tmp_path):
    clips = {f"images/{n}": [{"id": n, "file_name": f"images/{n}/0.bmp"}] for n in (1, 2, 3)}
    data = {"images": [frames[0] for frames in clips.values()], "annotations": []}
    release = Split(tmp_path, data, clips)
    source = tmp_path / "train.txt"
    source.write_text("/upstream/ITSDT/images/1/0.bmp\n/upstream/ITSDT/images/3/0.bmp 0,0,3,3,0\n")
    assert set(itsdt_development(release, source).clips) == {"images/1", "images/3"}
    source.write_text("/upstream/ITSDT/images/1/1.bmp\n")
    with pytest.raises(ValueError, match="disagree"):
        itsdt_development(release, source)


def test_multiclass_view_and_clip_local_order(tmp_path):
    train = make_clip(tmp_path, "train", "car")
    val = make_clip(tmp_path, "val", "car")
    test = make_clip(tmp_path, "test", "car")
    # The public XS-VID compatibility JSON can restart video IDs at zero.
    check_disjoint(train, val, test, video=True)
    assert [frame["id"] for frame in train.clips["images/train"]] == [1, 2, 10]
    with pytest.raises(ValueError, match="more than one split"):
        check_disjoint(train, train, test, video=True)
    args = SimpleNamespace(train_stride=1, val_stride=1, dataset="xsvid", short_gap=2, long_gap=12)
    root = tmp_path / "view"
    prepare_view(root, {"train": train, "val": val}, train.coco["categories"][:1], "long", True, args)
    assert (root / "labels/train/1.txt").read_text().split() == [
        "0", "0.250000000", "0.250000000", "0.125000000", "0.125000000"
    ]
    assert (root / "labels/train/2.txt").read_text() == ""
    with Image.open(root / "images/train/1.jpg") as image:
        assert all(abs(channel - 114) <= 3 for channel in image.getpixel((35, 35)))
        assert image.getpixel((16, 16))[1] < 10


def test_prediction_uses_each_splits_category_ids(tmp_path, monkeypatch):
    import torch
    import small_object_benchmark

    split = make_clip(tmp_path, "test", "car", category_id=9)
    args = SimpleNamespace(dataset="xsvid", predict_batch=2)
    box = SimpleNamespace(xyxy=torch.tensor([[1, 2, 4, 6]]), conf=torch.tensor([0.75]), cls=torch.tensor([0]))
    monkeypatch.setattr(small_object_benchmark, "predict_batch",
                        lambda model, sources, args: [SimpleNamespace(boxes=box) for _ in sources])
    predictions = predict(split, None, [{"id": 2, "name": "car"}], "rgb", False, args)
    assert [row["image_id"] for row in predictions] == [1, 2, 10]
    assert {row["category_id"] for row in predictions} == {9}
    assert predictions[0]["bbox"] == [1, 2, 3, 4]


def test_visdrone_submission_keeps_frame_and_category(tmp_path):
    folder = tmp_path / "sequences/clip"
    folder.mkdir(parents=True)
    labels = tmp_path / "annotations"
    labels.mkdir()
    Image.new("RGB", (64, 64)).save(folder / "0000001.jpg")
    (labels / "clip.txt").write_text("1,1,10,12,8,8,1,4,0,0\n1,0,25,25,8,8,0,0,0,0\n")
    split = visdrone_split(tmp_path)
    assert split.coco["categories"][3] == {"id": 4, "name": "car"}
    predictions = [{"image_id": 1, "category_id": 4, "bbox": [10, 12, 8, 8], "score": 0.9}]
    out = tmp_path / "submission"
    visdrone_predictions(split, predictions, out)
    assert (out / "clip.txt").read_text().strip().split(",")[7:] == ["4", "-1", "-1"]


def test_coco_scoring_with_empty_and_perfect_predictions(tmp_path):
    pytest.importorskip("pycocotools")
    split = make_clip(tmp_path, "test", "car")
    gt = split.coco["annotations"][0]
    prediction = {"image_id": 1, "category_id": 2, "bbox": gt["bbox"], "score": 1.0}
    assert score_coco(split.coco, [], "irdst-r")["AP50"] == 0
    assert score_coco(split.coco, [prediction], "irdst-r")["AP50"] == pytest.approx(100)


def test_checkpoint_category_matching_rejects_unscored_classes():
    from small_object_benchmark import checkpoint_category_ids

    categories = [{"name": "ignore", "id": 4}, {"name": "Car", "id": 2}, {"name": "boat", "id": 6}]
    assert checkpoint_category_ids({0: "boat", 1: "car", 2: "bird"}, categories, False) == {0: 6, 1: 2, 2: None}
    with pytest.raises(ValueError, match="missing benchmark classes: boat, car"):
        checkpoint_category_ids({0: "uav"}, categories, False)
    with pytest.raises(ValueError, match="one scored benchmark class"):
        checkpoint_category_ids({0: "uav"}, categories, True)
    assert checkpoint_category_ids({0: "uav"}, categories[:2], True) == {0: 2}


def test_inference_pilot_reads_only_staged_test_clip(tmp_path):
    from small_object_benchmark import load_test_split

    release = tmp_path / "XS-VID-v2"
    clip = make_clip(release, "a-clip", "car")
    clip.coco["images"].append({"id": 99, "file_name": "z-unstaged/0000001.jpg", "width": 64, "height": 64})
    annotation = release / "annotations/protocols/paper_detection_test.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(json.dumps(clip.coco))
    args = SimpleNamespace(dataset="xsvid", data_root=tmp_path, test_json=None, stage="predict", limit=1)
    test, provenance = load_test_split(args)
    assert list(test.clips) == ["images/a-clip"]
    assert len(test.coco["images"]) == 3
    assert provenance["test_json"] == str(annotation)
    args.stage, args.limit = "score", None
    test, _ = load_test_split(args)
    assert len(test.coco["images"]) == 4
    assert test.clips == {}
