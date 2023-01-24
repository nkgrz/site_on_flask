# import io
# import cv2
# import requests
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from skimage.transform import resize
# from requests.auth import HTTPBasicAuth
# from keras.models import model_from_json
# from werkzeug.utils import secure_filename
from flask import Flask, render_template, request


# немного про запросы
# https://docs-python.ru/packages/veb-frejmvork-flask-python/dostup-razlichnym-dannym-zaprosa-flask/


app = Flask(__name__)


@app.route('/')
@app.route('/home')
def index():
    return render_template('home.html')


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

if __name__ == '__main__':
    app.run(debug=True)