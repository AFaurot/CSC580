import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Fix for runtime error tf.placeholder() is not compatible with eager execution.
tf.compat.v1.disable_eager_execution()

# In order to make the random numbers predictable, define fixed seeds
np.random.seed(101)

# I had to change this line from assignment template for compatibility with TF 2.x
# I am using Tensorflow 2.19.0
# Original line: tf.set_random_seed(101)
# Results in error: AttributeError: module 'tensorflow' has no attribute 'set_random_seed'
tf.compat.v1.set_random_seed(101)


# Generate random linear data

x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

# Add noise
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

n = len(x) # number of data points
print("Number of data points: {}".format(n))

# 1) Plot the training data
plt.figure()
plt.scatter(x, y)
plt.title("Training Data (Noisy Linear)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 2) Create placeholders X and Y
X = tf.compat.v1.placeholder(tf.float32, name="X")
Y = tf.compat.v1.placeholder(tf.float32, name="Y")

# 3) Trainable variables: weight and bias
W = tf.Variable(np.random.randn(), dtype=tf.float32, name="weight")
b = tf.Variable(np.random.randn(), dtype=tf.float32, name="bias")

# 4) Hyperparameters
learning_rate = 0.01
training_epochs = 1000

# 5) Hypothesis, cost function, optimizer

# Hypothesis (prediction)
y_pred = W * X + b

# Cost function (Mean Squared Error)
cost = tf.reduce_mean(tf.square(y_pred - Y))

# Optimizer (Adam Optimizer)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 6) Training inside a session
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    # Training loop
    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={X: x, Y: y})

        # print progress occasionally
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{training_epochs}, cost={c:.4f}")

    # 7) Print results
    final_cost = sess.run(cost, feed_dict={X: x, Y: y})
    final_W = sess.run(W)
    final_b = sess.run(b)

    print("\nTraining complete.")
    print(f"Final cost (MSE): {final_cost:.4f}")
    print(f"Learned weight (W): {final_W:.4f}")
    print(f"Learned bias (b): {final_b:.4f}")

    # 8) Plot fitted line on top of original data
    fitted_y = final_W * x + final_b

plt.figure()
plt.scatter(x, y, label="Training data")
plt.plot(x, fitted_y, label="Fitted line")
plt.title("Linear Regression Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()