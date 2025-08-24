import base64

img_path = "resc-muscle-img.png"
output = "resc-muscle-imgencoded.txt"

with open(img_path,"rb") as file:
    encoded = base64.b64encode(file.read())

    with open(output,"w") as outFile:
        outFile.write(encoded.decode("utf-8"))