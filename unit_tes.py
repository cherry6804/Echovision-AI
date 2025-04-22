import unittest
import os
from echo import EchoNetModel

class TestEchoNetModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize model with actual dataset path"""
        cls.dataset_video_path = "C:\\Users\\Admin\\Downloads\\EchoNet-Dynamic\\Videos\\0X100009310A3BD7FC.avi"  # Change this to a real file from your dataset

        if not os.path.exists(cls.dataset_video_path):
            raise FileNotFoundError(f"Test video file not found: {cls.dataset_video_path}")

        cls.model = EchoNetModel(
            csv_path="C:\\Users\\Admin\\Downloads\\EchoNet-Dynamic\\FileList.csv",
            volume_csv_path="C:\\Users\\Admin\\Downloads\\EchoNet-Dynamic\\VolumeTracings.csv",
            video_dir="C:\\Users\\Admin\\Downloads\\EchoNet-Dynamic\\Videos\\"
        )

    def test_preprocess_video(self):
        """Test video preprocessing for prediction"""
        processed = self.model.preprocess_video(self.dataset_video_path)
        self.assertEqual(processed.shape, (5, 64, 64, 3))

    def test_prediction_structure(self):
        """Test prediction format"""
        result = self.model.predict_video(self.dataset_video_path)
        self.assertIn("predictions", result)
        self.assertIn("average_ef", result)
        self.assertIn("risk", result)

# Custom Test Result to modify output
class CustomTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        self.stream.writeln(f"{test} ... ✅ Passed")

# Custom Test Runner to use the modified result format
class CustomTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return CustomTestResult(self.stream, self.descriptions, self.verbosity)

if __name__ == "__main__":
    print("\nRunning Test Cases...\n")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestEchoNetModel)
    runner = CustomTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n🎉 All Test Cases Passed Successfully! ✅")
    else:
        print("\n❌ Some Test Cases Failed. Check logs for errors.")
