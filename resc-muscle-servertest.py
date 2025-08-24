import requests


with open("resc-muscle-imgencoded.txt", "r") as f:
    img64 = f.read()

print(img64[0:10])

payload = {'image': "hi"}
response_form = requests.post('http://127.0.0.1:5000/api/draftMessage-progress', data=payload)

chat = response_form.text
print(chat)