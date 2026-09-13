"""TensorRT placeholder labels must become canonical before policy handling."""

import unittest

import numpy as np

from rams.models import COCO_NAMES, ModelWrapper, Tier
from rams.policy import make_policy


class _Array:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _Box:
    def __init__(self, cls_id, confidence, xyxy):
        self.cls = cls_id
        self.conf = confidence
        self.xyxy = _Array(xyxy)


class _Result:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeEngine:
    def __init__(self, names, boxes):
        self.names = names
        self._boxes = boxes

    def __call__(self, frame, verbose=False, imgsz=None):
        return [_Result(self._boxes, self.names)]


PLACEHOLDER_NAMES = {index: f"class{index}" for index in range(100)}
PERSON_BOX = [100.0, 60.0, 240.0, 300.0]
CAR_BOX = [620.0, 179.0, 700.0, 238.0]


def _infer(tier, names, boxes, backend="tensorrt"):
    wrapper = ModelWrapper(tier)
    wrapper._backend = backend
    wrapper._loaded = True
    wrapper._model = _FakeEngine(names, boxes)
    frame = np.zeros((376, 1241, 3), dtype="uint8")
    return wrapper.infer(frame)["detections"]


class TensorRtClassLabelTests(unittest.TestCase):
    def test_placeholder_engine_ids_become_canonical_labels(self):
        detections = _infer(Tier.NANO, PLACEHOLDER_NAMES, [_Box(5, 0.9, CAR_BOX), _Box(2, 0.7, CAR_BOX)])
        self.assertEqual([detection["class"] for detection in detections], ["bus", "car"])
        self.assertTrue(all(detection["class"] in COCO_NAMES for detection in detections))

    def test_canonical_person_reaches_two_level_policy(self):
        detections = _infer(Tier.MEDIUM, PLACEHOLDER_NAMES, [_Box(0, 0.9, PERSON_BOX)])
        self.assertEqual(detections[0]["class"], "person")
        policy = make_policy("safety2")
        policy.observe(detections)
        self.assertEqual(policy.select_tier(0.1, Tier.NANO), Tier.MEDIUM)

    def test_non_placeholder_metadata_and_non_tensorrt_path_are_unchanged(self):
        named = _infer(Tier.NANO, {2: "car"}, [_Box(2, 0.9, CAR_BOX)])
        self.assertEqual(named[0]["class"], "car")
        pytorch = _infer(Tier.NANO, PLACEHOLDER_NAMES, [_Box(2, 0.9, CAR_BOX)], backend="ultralytics")
        self.assertEqual(pytorch[0]["class"], "class2")

    def test_out_of_range_placeholder_degrades_to_id(self):
        detection = _infer(Tier.NANO, PLACEHOLDER_NAMES, [_Box(99, 0.9, CAR_BOX)])[0]
        self.assertEqual(detection["class"], "99")


if __name__ == "__main__":
    unittest.main()
