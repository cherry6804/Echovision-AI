import unittest
import requests
import time

BASE_URL = "http://127.0.0.1:5000"
FILE_PATH = "C:\\Users\\Admin\\Downloads\\EchoNet-Dynamic\\Videos\\0X100009310A3BD7FC.avi"
def wait_for_server():
    """Wait for the Flask server to start before running tests"""
    for _ in range(10):  # Try for 10 seconds
        try:
            response = requests.get(BASE_URL)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    return False
class TestSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Ensure Flask server is running before testing."""
        if not wait_for_server():
            raise RuntimeError("Flask server is not running. Start the server before testing.")

    def test_home_page(self):
        """Check if the homepage is loading correctly"""
        response = requests.get(BASE_URL + "/")
        self.assertEqual(response.status_code, 200)

    def test_open_predict_page(self):
        """Check if prediction page is loading"""
        response = requests.get(BASE_URL + "/open_predict")
        self.assertEqual(response.status_code, 200)

    def test_video_upload(self):
        """Test video upload and prediction"""
        with open(FILE_PATH, 'rb') as f:
            files = {'file': f}
            response = requests.post(BASE_URL + "/predict", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.json())

    @classmethod
    def tearDownClass(cls):
        """Print success message after all tests pass"""
        print("\n✅ All test cases passed successfully!")

if __name__ == "__main__":
    unittest.main()
