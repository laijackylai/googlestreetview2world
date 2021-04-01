import os
from dotenv import load_dotenv
import google_streetview.api

load_dotenv()

API_KEY = os.environ.get('API_KEY')

params = [{
    'size': '640x640',
    'location': '22.269785,114.1300342',
    'heading': '0;90;180;270',
    'pitch': '0',
    'key': API_KEY
}]

results = google_streetview.api.results(params)

results.preview()
results.download_links('raw_img')
