import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# === Функции анализа изображения ===
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return img, edges

def extract_graph_data(edges):
    # Используем контуры для выделения линий графика
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    data_points = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            data_points.append((x, y))
    return np.array(data_points)

# === Прогнозирование ===
def predict_trend(data_points, steps=50):
    x = data_points[:, 0].reshape(-1, 1)
    y = data_points[:, 1]
    model = LinearRegression()
    model.fit(x, y)
    future_x = np.arange(x.max() + 1, x.max() + steps).reshape(-1, 1)
    future_y = model.predict(future_x)
    return future_x.flatten(), future_y

# === Интерфейс ===
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        image_path = "path_to_graph_image.jpg"
        file.save(image_path)

        img, edges = preprocess_image(image_path)
        data_points = extract_graph_data(edges)
        future_x, future_y = predict_trend(data_points)

        plt.figure()
        plt.scatter(data_points[:, 0], data_points[:, 1], label="Original Data", color="blue")
        plt.plot(future_x, future_y, label="Predicted Data", color="red")
        plt.legend()
        plt.savefig("static/result.png")
        return render_template('result.html', graph_url="static/result.png")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
