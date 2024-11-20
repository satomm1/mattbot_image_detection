import requests
import shutil

url='http://127.0.0.1:5000/gemini'
query = 'Provide the name of the most prominent object in this image.'

image_name = 'unknown_object.jpg'

# Copy image to /gemini_code/
shutil.copyfile(image_name, f'../../../../../gemini_code/{image_name}')


data = {'query': query, 'query_type': 'image', 'image_name': image_name}
response = requests.post(url, json=data)
result = response.json()
print(result['response'])