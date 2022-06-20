import unittest

from trajdata import UnifiedDataset


class TestDescriptionMatching(unittest.TestCase):
    def test_night(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini"], scene_description_contains=["night"]
        )

        for scene_info in dataset.scenes():
            self.assertIn("night", scene_info.description)

    def test_intersection(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini"], scene_description_contains=["intersection"]
        )

        for scene_info in dataset.scenes():
            self.assertIn("intersection", scene_info.description)

    def test_intersection_more_initial(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "lyft_sample"],
            scene_description_contains=["intersection"],
        )

        for scene_info in dataset.scenes():
            self.assertIn("intersection", scene_info.description)


if __name__ == "__main__":
    unittest.main()
