from django.shortcuts import render
from imageio import imread
from PIL import Image #for image.resize
from tensorflow.compat.v1.keras.backend import set_session
import numpy as np
import re
import sys

##mnist model path
import os
sys.path.append(os.path.abspath("./model"))

##custom utils file for writing helper function
from main.utils import init

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

#declare global variables
global model, graph, session

model, graph, session = init()

import base64
from io import BytesIO

#declare output path for our image
OUTPUT = os.path.join(os.path.dirname(__file__), 'output.png')

def getI420FromBase64(codec):
    base64_data = re.sub('^data:image/.+;base64,', '', codec)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    img.save(OUTPUT)
    
def convertImage(imgData):
    getI420FromBase64(imgData)

'''
main API:
Grab a base64 image file submitted by client
Convert it into png file
Process it to be able to fit in our trained model file
Predict the image using our previous helper function and get performance metric in return
Return it as a JSON response
'''

@csrf_exempt
def predict(request):
    print('yes')
    imgData = request.POST.get('img')
    convertImage(imgData)
    
    x = imread(OUTPUT, pilmode='L')
    x = np.invert(x)
    x = np.array(Image.fromarray(x).resize((28,28)))
    x = x.reshape(1,28,28)
    
    with graph.as_default():
        set_session(session)
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        response = np.array_str(np.argmax(out, axis=1))
        print(response[1])
        return JsonResponse({"ouput": response})
    
def index(request):
    return render(request, 'index2.html',{})

