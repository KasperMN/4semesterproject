import unittest
import requests

from API import Controller


class TestController(unittest.TestCase):
    def test_get_request(self):
        """
            Should return
        """
        api_url = "http://127.0.0.1:5000/data/https://link.speckle.dk/api/objects/5f57e166f2ebf23b47b317af"

        Controller.start_api()  # Running api server
        response = requests.get(url=api_url)  # Calling get on api

        self.assertEqual(response.status_code, "200")
        self.assertIsNotNone(response)

    def test(self):
        self.assertIs("hej", "hej")


if __name__ == '__main__':
    unittest.main()
