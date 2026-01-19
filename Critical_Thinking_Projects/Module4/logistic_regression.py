import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Disable eager execution for TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

# Set random seed for repeatability
np.random.seed(0)

# Number of samples
N = 100

# Zeros form a Gaussian centered at (-1, -1)
x_zeros = np.random.multivariate_normal(
    mean=np.array((-1, -1)),
    cov=0.1 * np.eye(2),
    size=(N // 2,)
)
y_zeros = np.zeros((N // 2,))

# Ones form a Gaussian centered at (1, 1)
x_ones = np.random.multivariate_normal(
    mean=np.array((1, 1)),
    cov=0.1 * np.eye(2),
    size=(N // 2,)
)
y_ones = np.ones((N // 2,))

x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])

# Plot the synthetic data (x_zeros and x_ones)
plt.figure()
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], label="Class 0", alpha=0.8)
plt.scatter(x_ones[:, 0], x_ones[:, 1], label="Class 1", alpha=0.8)
plt.title("Synthetic Data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()

# Placeholders for input data and labels
with tf.name_scope("placeholders"):
    x = tf.compat.v1.placeholder(tf.float32, (N, 2))
    y = tf.compat.v1.placeholder(tf.float32, (N,))

with tf.name_scope("weights"):
    W = tf.Variable(tf.random.normal((2, 1)))
    b = tf.Variable(tf.random.normal((1,)))

# Logistic regression prediction
with tf.name_scope("prediction"):
    y_logit = tf.squeeze(tf.matmul(x, W) + b)
    # the sigmoid gives the class probability of 1
    y_one_prob = tf.sigmoid(y_logit)
    # Rounding P(y=1) will give the correct prediction.
    y_pred = tf.round(y_one_prob)

with tf.name_scope("loss"):
    # Compute the cross-entropy term for each datapoint
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y_logit,
        labels=y
    )
    # Sum all contributions
    l = tf.reduce_sum(entropy)

# Adam optimizer
with tf.name_scope("optim"):
    train_op = tf.compat.v1.train.AdamOptimizer(0.01).minimize(l)

# TensorBoard summaries
with tf.name_scope("summaries"):
    tf.compat.v1.summary.scalar("loss", l)
    merged = tf.compat.v1.summary.merge_all()

train_writer = tf.compat.v1.summary.FileWriter(
    "logistic-train",
    tf.compat.v1.get_default_graph()
)

# Train the model
num_steps = 2000
print_every = 200

# Plot loss graph to show in paper
loss_history = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(1, num_steps + 1):
        feed = {
            x: x_np.astype(np.float32),
            y: y_np.astype(np.float32)
        }
        _, loss_val, summary = sess.run(
            [train_op, l, merged],
            feed_dict=feed
        )

        train_writer.add_summary(summary, global_step=step)
        loss_history.append(loss_val)

        if step % print_every == 0:
            print(f"Step {step:4d}  Loss: {loss_val:.4f}")

    # Retrieve trained parameters
    W_val, b_val = sess.run([W, b])

train_writer.close()

print("\nTrained W:", W_val.ravel())
print("Trained b:", b_val)

# Plot loss history
plt.figure()
plt.plot(loss_history)
plt.title("Loss History")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Compute decision boundary: W1*x1 + W2*x2 + b = 0
W1, W2 = float(W_val[0, 0]), float(W_val[1, 0])
b0 = float(b_val[0])

plt.figure()
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], label="True Class 0", alpha=0.8)
plt.scatter(x_ones[:, 0], x_ones[:, 1], label="True Class 1", alpha=0.8)

x1_min = x_np[:, 0].min() - 0.5
x1_max = x_np[:, 0].max() + 0.5
x1_line = np.linspace(x1_min, x1_max, 200)

# Solve for x2 = -(W1*x1 + b) / W2
if abs(W2) < 1e-8:
    x1_boundary = -b0 / W1
    plt.axvline(x=x1_boundary, linestyle="--", label="Decision boundary")
else:
    x2_line = -(W1 * x1_line + b0) / W2
    plt.plot(x1_line, x2_line, linestyle="--", label="Decision boundary")

plt.title("True Data with Learned Decision Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()
