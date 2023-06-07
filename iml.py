import numpy as np

def locally_weighted_regression(X, y, query_point, tau):
    m = X.shape[0]  # number of training examples
    X = np.hstack((np.ones((m, 1)), X))  # add a column of ones to X

    # Initialize weights to 1
    weights = np.eye(m)

    # Calculate weights for each training example
    for i in range(m):
        xi = X[i]
        xq = query_point
        weights[i, i] = np.exp(np.dot((xi - xq), (xi - xq).T) / (-2 * tau * tau))

    # Calculate theta using normal equation
    theta = np.linalg.pinv(X.T.dot(weights).dot(X)).dot(X.T).dot(weights).dot(y)

    # Calculate the predicted value
    query_point = np.hstack((1, query_point))  # add a one to the query point
    y_pred = np.dot(query_point, theta)

    return y_pred

# Example usage
X = np.array([[1, 1], [2, 3], [4, 3], [3, 6], [5, 8]])
y = np.array([1, 2, 3, 4, 5])
query_point = np.array([3, 4])
tau = 1.0

prediction = locally_weighted_regression(X, y, query_point, tau)
print("Predicted value:", prediction)