import json
import requests


def get_config():
    with open('ebird_config.json') as f:
        data = json.load(f)
        apis = data['apis']
        ebird_token = data['ebird_token']
        return ebird_token, apis


def get_region_recent_sightings(country, region, days_back=2, max_results=100):
    ebird_token, apis = get_config()
    url = apis['region_recent']
    headers = {'X-eBirdApiToken': ebird_token}
    response = requests.get(url.format(country=country, region=region, days_back=days_back, max_results=max_results),
                            headers=headers)
    return response.json()
