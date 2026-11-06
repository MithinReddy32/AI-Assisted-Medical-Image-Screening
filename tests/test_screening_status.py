import unittest

from XrayScreening_V2.src.screening_status import non_chest_status


class ScreeningStatusTests(unittest.TestCase):
    ready = {"status": "ready", "specialist": "mura"}

    def test_positive_and_threshold_boundary_recommend_doctor_review(self):
        for score in [.5, .8]:
            result = non_chest_status("Wrist", self.ready, score)
            self.assertEqual(result["title"], "Possible abnormality — consult a doctor")

    def test_below_threshold_does_not_claim_image_is_normal(self):
        result = non_chest_status("Hand", self.ready, .2)
        self.assertEqual(result["title"], "No abnormality flagged")
        self.assertIn("cannot confirm", result["message"])

    def test_unsupported_body_part_does_not_get_a_normal_result(self):
        result = non_chest_status("Knee", {"status": "unsupported", "specialist": None})
        self.assertEqual(result["title"], "Unable to assess — consult a doctor")
        self.assertIn("not available for Knee", result["message"])

    def test_deferred_route_does_not_use_an_abnormality_score(self):
        for status in ["uncertain", "ambiguous", "unavailable"]:
            result = non_chest_status("Wrist", {"status": status, "specialist": None}, .1)
            self.assertEqual(result["title"], "Unable to assess — consult a doctor")

    def test_missing_or_invalid_inference_does_not_get_a_normal_result(self):
        for score in [None, float("nan"), float("inf"), -.1, 1.1]:
            result = non_chest_status("Elbow", self.ready, score)
            self.assertEqual(result["title"], "Unable to assess — consult a doctor")

    def test_chest_does_not_get_an_upper_limb_result(self):
        self.assertIsNone(non_chest_status("Chest", {"status": "ready", "specialist": "chest"}))


if __name__ == "__main__":
    unittest.main()
