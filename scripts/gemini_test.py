import requests
import shutil

url='http://127.0.0.1:5000/gemini'
query = 'Provide a list of the objects in this picture?'

image_name = 'homography_image.jpg'

# Copy image to /gemini_code/
shutil.copyfile(image_name, f'../../../../../gemini_code/{image_name}')


data = {'query': query, 'image_name': 'homography_image.jpg'}
response = requests.post(url, json=data)
result = response.json()
print(result['response'])