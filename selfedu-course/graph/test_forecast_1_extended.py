import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Step 1:
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return img, binary

# Step 2:
def extract_graph_data(binary_image):
    #
    points = np.column_stack(np.where(binary_image > 0))

    #
    clustering = DBSCAN(eps=5, min_samples=10).fit(points)
    labels = clustering.labels_

    #
    unique_labels, counts = np.unique(labels, return_counts=True)
    main_cluster_label = unique_labels[np.argmax(counts[1:]) + 1] #
    main_points = points[labels == main_cluster_label]

    #
    sorted_points = main_points[np.argsort(main_points[:, 1])]
    x, y = sorted_points[:, 1], sorted_points[:, 0]
    return x, y

#
def predict_future(x, y, steps=50):
    #
    poly = PolynomialFeatures(degree=4)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly, y)

    #
    future_x = np.linspace(x.max() + 1, x.max() + steps, steps)
    future_x_poly = poly.transform(future_x.reshape(-1, 1))
    future_y = model.predict(future_x_poly)
    return future_x, future_y

#
def plot_graph(original_x, original_y, future_x, future_y):
    plt.figure(figsize=(10, 6))
    plt.scatter(original_x, original_y, label='Original Data', color='blue', s=10)
    plt.plot(future_x, future_y, label='Predicted Data', color='red', linewidth=2)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Graph Analysis and Prediction")
    plt.legend()
    plt.grid(True)
    plt.show()

#
def main():
    image_path = 'img_4.png'
    img, binary = load_and_preprocess_image(image_path)
    x, y = extract_graph_data(binary)
    future_x, future_y = predict_future(x, y)
    plot_graph(x, y, future_x, future_y)

if __name__ == "__main__":
    main()
