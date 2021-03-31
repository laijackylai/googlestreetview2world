import os
from dotenv import load_dotenv
import google_streetview.api

load_dotenv()

API_KEY = os.environ.get('API_KEY')

params = [{
    'size': '640x640',
    'location': '22.2694989,114.1304372',
    'heading': '90',
    'pitch': '0',
    'key': API_KEY
}]

results = google_streetview.api.results(params)

results.preview()
results.download_links('raw_img')
