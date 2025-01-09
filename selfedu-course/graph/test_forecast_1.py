import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Load image and preprocess
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return img, binary

# Step 2: Extract graph data
def extract_graph_data(binary_image):
    # Assuming graph is white on black
    points = np.column_stack(np.where(binary_image > 0))
    # Convert coordinates from pixel to user system
    x, y = points[:, 1], points[:, 0]
    return x, y

# Step 3: Predict continuation
def predict_future(x, y, steps=50):
    # Polynomial regression for prediction
    poly = PolynomialFeatures(degree=3)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly, y)
    # Generate new values
    future_x = np.linspace(x.max() + 1, x.max() + steps, steps)
    future_x_poly = poly.transform(future_x.reshape(-1, 1))
    future_y = model.predict(future_x_poly)
    return future_x, future_y

# Step 4: Visualization
def plot_graph(x, y, future_x, future_y):
    plt.scatter(x, y, label="Original Data", color="blue")
    plt.plot(future_x, future_y, label="Predicted Data", color="red")
    plt.legend()
    plt.show()

# Main process
image_path = 'path_to_graph_image.jpg'
img, binary = load_and_preprocess_image(image_path)
x, y = extract_graph_data(binary)
future_x, future_y = predict_future(x, y)
plot_graph(x, y, future_x, future_y)