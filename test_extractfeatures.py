import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import requests
from typing import Iterator

# Make sure the script can find the extractfeatures module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from extractfeatures import get_data_stream, preprocess_image

class TestDataStream(unittest.TestCase):
    """Unit tests for the data streaming functionality in extractfeatures.py."""

    def setUp(self):
        """Set up common test data and mocks."""
        # Create a dummy CSV in memory for the test
        self.dummy_csv_data = "image_url,label\nhttp://example.com/fake.jpg,1\nhttp://example.com/real.jpg,0"
        
        # Mock pandas to read our dummy CSV data
        self.mock_pd_read_csv = patch('pandas.read_csv', return_value=pd.read_csv(pd.io.common.StringIO(self.dummy_csv_data)))
        self.mock_pd_read_csv.start()

        # Mock glob to find a "file"
        self.mock_glob = patch('glob.glob', return_value=['dummy/path/to/data.csv'])
        self.mock_glob.start()

        # Mock requests.get to return a fake response
        self.mock_requests_get = patch('requests.get')
        self.mock_get = self.mock_requests_get.start()
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.content = b'fake image bytes'
        self.mock_get.return_value = mock_response

        # Mock nfstream to avoid network capture
        self.mock_nfstream = patch('nfstream.NFStreamer', autospec=True)
        self.mock_nfstream.start()
        
        # Mock socket.gethostbyname
        self.mock_gethostbyname = patch('socket.gethostbyname', return_value='127.0.0.1')
        self.mock_gethostbyname.start()

        # Mock time.sleep to speed up tests
        self.mock_sleep = patch('time.sleep', return_value=None)
        self.mock_sleep.start()

    def tearDown(self):
        """Stop all patches."""
        patch.stopall()

    def test_get_data_stream_is_generator(self):
        """Test that get_data_stream returns a generator."""
        stream = get_data_stream()
        self.assertIsInstance(stream, Iterator, "get_data_stream should return an iterator/generator.")

    def test_stream_yields_correct_structure(self):
        """Test that the yielded record has the correct keys and data types."""
        # Get the first record from the stream
        stream = get_data_stream()
        data_record = next(stream)

        self.assertIsInstance(data_record, dict, "Stream should yield dictionaries.")
        
        # Check for the presence of all expected keys
        expected_keys = [
            "url", "label", "dst_ip", "http_status", "content_length_bytes",
            "response_time_s", "image_data", "flow_features"
        ]
        for key in expected_keys:
            self.assertIn(key, data_record, f"Key '{key}' should be in the data record.")
            
        # Check data types of a few key fields
        self.assertIsInstance(data_record['url'], str)
        self.assertIsInstance(data_record['http_status'], int)
        self.assertIsNotNone(data_record['image_data'])

    def test_preprocess_image_returns_bytes(self):
        """Test the placeholder preprocess_image function."""
        test_bytes = b'test_image'
        processed = preprocess_image(test_bytes)
        self.assertEqual(processed, test_bytes, "preprocess_image should return the bytes it was given.")

    def test_stream_handles_request_exception(self):
        """Test that the stream continues gracefully if a request fails."""
        # Configure the mock to raise an exception
        self.mock_get.side_effect = requests.exceptions.RequestException("Test error")

        stream = get_data_stream()
        data_record = next(stream) # Get the first record

        # Check that the record reflects the failure
        self.assertEqual(data_record['http_status'], -1, "HTTP status should be -1 on error.")
        self.assertEqual(data_record['content_length_bytes'], 0, "Content length should be 0 on error.")
        self.assertIsNone(data_record['image_data'], "Image data should be None on error.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
