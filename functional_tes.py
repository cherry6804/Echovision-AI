import unittest
import json
import requests

BASE_URL = "http://127.0.0.1:5000"

class TestFunctional(unittest.TestCase):

    def test_no_file_upload(self):
        """Test API response for no file uploaded"""
        response = requests.post(BASE_URL + "/predict", files={})
        self.assertEqual(response.status_code, 200)
        self.assertIn("error", response.json())

    def test_invalid_file_upload(self):
        """Test API response for invalid file format"""
        with open("test_invalid.txt", "w") as f:
            f.write("This is not a video file")
        with open("test_invalid.txt", 'rb') as f:
            files = {'file': f}
            response = requests.post(BASE_URL + "/predict", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn("error", response.json())

if __name__ == "__main__":
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestFunctional))
    if result.wasSuccessful():
        print("✅ All test cases passed successfully!")
