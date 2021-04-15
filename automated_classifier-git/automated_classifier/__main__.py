from automated_classifier.api import run_restful_api
import requests

if __name__ == '__main__':
    run_restful_api()  # Start the api

    all_objects = {
        "keys": ["Area", "Volume", "Assembly Code"],
        "url": "https://link.speckle.dk/api/streams/grB5WJmsE/objects",
        "target": "Assembly Code",
        "table_name": "Walls"
    }

    single_object = {
        "keys": ["Area", "Volume", "Assembly Code"],
        "url": "https://link.speckle.dk/api/objects/5f57e166f2ebf23b47b317af",
        "target": "Assembly Code",
        "table_name": "Walls"
    }

    data = requests.post(url="http://127.0.0.1:5000/data", data=single_object).json()
    print(data)

