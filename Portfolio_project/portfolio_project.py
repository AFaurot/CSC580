import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Environment / reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Better GPU behavior (I am using WSL + NVIDIA 4070 in Windows 11 env)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# Step 2: Load CIFAR-10 data
(X, y), (_, _) = keras.datasets.cifar10.load_data()

# Selecting a single class of images (0-9). Target class of 8 chosen to match the template code
target_class = 8
X = X[y.flatten() == target_class]

# Normalize the input
X = (X.astype(np.float32) / 127.5) - 1.0

# Step 3: Define parameters
image_shape = (32, 32, 3)
latent_dimensions = 100

num_steps = 15000
batch_size = 32
display_interval = 2500

output_dir = "gan_outputs"
os.makedirs(output_dir, exist_ok=True)

# Added a function  to show the real samples for comparison
def show_real_samples(X, n=16):
    idx = np.random.randint(0, X.shape[0], n)
    imgs = X[idx]
    imgs = 0.5 * imgs + 0.5  # back to [0,1]
    r = c = int(np.sqrt(n))
    fig, axs = plt.subplots(r, c, figsize=(6, 6))
    k = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(imgs[k])
            axs[i, j].axis("off")
            k += 1
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# Step 4: Build Generator - Matches the architecture from template code
def build_generator():
    model = keras.Sequential(name="generator")

    # Input layer
    model.add(layers.Dense(128 * 8 * 8, activation="relu", input_shape=(latent_dimensions,)))
    model.add(layers.Reshape((8, 8, 128)))

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.78))
    model.add(layers.Activation("relu"))

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.78))
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(3, kernel_size=3, padding="same"))
    model.add(layers.Activation("tanh"))

    # Generating the output image
    noise = layers.Input(shape=(latent_dimensions,))
    image = model(noise)
    return keras.Model(noise, image, name="generator_model")


# Step 5: Build Discriminator - Matches the architecture from template code
def build_discriminator():
    # Building the convolutional layers to classify real vs fake images
    model = keras.Sequential(name="discriminator")

    model.add(layers.Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(layers.BatchNormalization(momentum=0.82))
    model.add(layers.LeakyReLU(negative_slope=0.25))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.82))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(negative_slope=0.25))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    image = layers.Input(shape=image_shape)
    validity = model(image)
    return keras.Model(image, validity, name="discriminator_model")


# Step 6: Utility to save generated image grids to disk
# I decided to save to disk to make them easier to share in the report.
# plt.show() is still called to display the images on screen during training.
def save_image_grid(generator, step, filename, r=4, c=4):
    noise = np.random.normal(0, 1, (r * c, latent_dimensions)).astype(np.float32)
    generated_images = generator(noise, training=False).numpy()

    # Scale from [-1, 1] -> [0, 1]
    generated_images = 0.5 * generated_images + 0.5
    generated_images = np.clip(generated_images, 0.0, 1.0)

    fig, axs = plt.subplots(r, c, figsize=(6, 6))
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(generated_images[count])
            axs[i, j].axis("off")
            count += 1
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    # Wrapping plt.show() in try catch to avoid issues in headless environments (like WSL without X server)
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display image grid on screen: {e}")
        pass
    plt.close(fig)


# Step 7: Build the GAN
generator = build_generator()
discriminator = build_discriminator()

g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
d_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

bce = keras.losses.BinaryCrossentropy(from_logits=False)


# Step 8: Training step function with @tf.function for better performance.
# Example expanded upon from TensorFlow's DCGAN tutorial below:
# https://www.tensorflow.org/tutorials/generative/dcgan#train-the-model

# This is where the adversarial training happens.
# It will be optimized by TensorFlow's autograph and run much faster as a result.
# This is a very different approach than the template code where a combined_network was built with the following steps :

# 1. compile discrimnator (trainable)
# 2. set discriminator.trainable = False
# 3. compile generator (with discriminator as part of the model)
# 4. set discriminator.trainable = True again for training the discriminator
# 5. sometimes had to re-compile the discriminator after setting trainable = True again for it to work properly

# So instead of above approach, the new approach uses GradientTape to compute
# the gradients for both the generator and discriminator in one function
@tf.function
def train_step(real_images):
    batch_n = tf.shape(real_images)[0]
    noise = tf.random.normal((batch_n, latent_dimensions))

    # Adversarial ground truths
    valid = tf.ones((batch_n, 1))
    fake = tf.zeros((batch_n, 1))

    # Same noise added from template code in assignment
    valid = valid + 0.05 * tf.random.uniform(tf.shape(valid))
    fake = fake + 0.05 * tf.random.uniform(tf.shape(fake))

    # GradiantTape documentation for reference: https://www.tensorflow.org/api_docs/python/tf/GradientTape
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        generated_images = generator(noise, training=True)

        real_pred = discriminator(real_images, training=True)
        fake_pred = discriminator(generated_images, training=True)

        d_loss_real = bce(valid, real_pred)
        d_loss_fake = bce(fake, fake_pred)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        g_loss = bce(valid, fake_pred)

    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)

    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss, g_loss


# Dataset as a Tensorflow pipeline is also a change recommended over Numpy approach used in template
# from tensorflow documentation
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices for reference
dataset = tf.data.Dataset.from_tensor_slices(X)
dataset = dataset.shuffle(buffer_size=min(10000, X.shape[0]))
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.repeat().prefetch(tf.data.AUTOTUNE)

data_iter = iter(dataset)

loss_history = []

# Show real samples before training for comparison
print("Showing real samples from the dataset before training:")
show_real_samples(X)

# Main training loop. This will run for a fixed number of steps controlled by num_steps.
for step in range(num_steps):
    real_batch = next(data_iter)

    d_loss, g_loss = train_step(real_batch)
    loss_history.append((step, float(d_loss), float(g_loss)))

    # Save first-step images
    if step == 0:
        first_path = os.path.join(output_dir, "step_0_grid.png")
        save_image_grid(generator, step, first_path)
        print(f"Saved first-step image grid to: {first_path}")

    # Periodic image grids
    if (step + 1) % display_interval == 0:
        path = os.path.join(output_dir, f"step_{step}_grid.png")
        save_image_grid(generator, step, path)
        print(f"Step {step}: D_loss={float(d_loss):.4f}, G_loss={float(g_loss):.4f} | Saved: {path}")

# Save last-step images
last_path = os.path.join(output_dir, f"step_{num_steps - 1}_grid.png")
save_image_grid(generator, num_steps - 1, last_path)
print(f"Saved last-step image grid to: {last_path}")

# Loss curve plot
loss_arr = np.array(loss_history, dtype=np.float32)
plt.figure()
plt.plot(loss_arr[:, 0], loss_arr[:, 1], label="Discriminator loss")
plt.plot(loss_arr[:, 0], loss_arr[:, 2], label="Generator loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
loss_plot_path = os.path.join(output_dir, "loss_curves.png")
plt.savefig(loss_plot_path, dpi=200)
plt.close()
print(f"Saved loss curves to: {loss_plot_path}")
