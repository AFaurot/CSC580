import tensorflow as tf
import seaborn as sns
import numpy as np
from PIL import Image
import glob
from collections import defaultdict
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Step 1: Image loading and preprocessing
IMG_SIZE = (94, 125)

def pixels_from_path(file_path):
    im = Image.open(file_path)
    im = im.resize(IMG_SIZE)
    np_im = np.array(im)
    # matrix of pixel RGB values
    return np_im


shape_counts = defaultdict(int)
for i, cat in enumerate(glob.glob('cats/*')[:1000]):
    if i % 100 == 0:
        print(i)
    img_shape = pixels_from_path(cat).shape
    shape_counts[str(img_shape)] = shape_counts[str(img_shape)] + 1

shape_items = list(shape_counts.items())
shape_items.sort(key=lambda x: x[1])
shape_items.reverse()

# 10% of the data will automatically be used for validation
validation_size = 0.1
img_size = IMG_SIZE  # resize images to be 374x500 (most common shape)
num_channels = 3  # RGB
sample_size = 8192  # We'll use 8192 pictures (2**13)
pixels_from_path(glob.glob('cats/*')[5]).shape

SAMPLE_SIZE = 2048
print("loading training cat images...")
cat_train_set = np.asarray([pixels_from_path(cat) for cat in glob.glob('cats/*')[:SAMPLE_SIZE]])
print("loading training dog images...")
dog_train_set = np.asarray([pixels_from_path(dog) for dog in glob.glob('dogs/*')[:SAMPLE_SIZE]])

valid_size = 512
print("loading validation cat images...")
cat_valid_set = np.asarray([pixels_from_path(cat) for cat in glob.glob('cats/*')[-valid_size:]])
print("loading validation dog images...")
dog_valid_set = np.asarray([pixels_from_path(dog) for dog in glob.glob('dogs/*')[-valid_size:]])

x_train = np.concatenate([cat_train_set, dog_train_set])
labels_train = np.asarray([1 for _ in range(SAMPLE_SIZE)] + [0 for _ in range(SAMPLE_SIZE)])
# fix for keras expecting 2D labels
labels_train = labels_train.reshape(-1, 1).astype("float32")
x_valid = np.concatenate([cat_valid_set, dog_valid_set])
labels_valid = np.asarray([1 for _ in range(valid_size)] + [0 for _ in range(valid_size)])
# fix for keras expecting 2D labels
labels_valid = labels_valid.reshape(-1, 1).astype("float32")

total_pixels = img_size[0] * img_size[1] * 3
fc_size = 512

inputs = keras.Input(shape=(img_size[1], img_size[0], 3), name='ani_image')
x = layers.Flatten(name='flattened_img')(inputs)  # turn image to vector.

x = layers.Dense(fc_size, activation='relu', name='first_layer')(x)
outputs = layers.Dense(1, activation='sigmoid', name='class')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

''' Step 2
. Use the AdamOptimizer,
. Use 10 epochs,
. Shuffle the training data
. Use MSE as the loss function.'''
customAdam = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=customAdam,  # Optimizer
              # Loss function to minimize
              loss="mean_squared_error",
              # List of metrics to monitor
              metrics=["binary_crossentropy", "mean_squared_error"])

print('# Fit model on training data for step 2: baseline fully connected network')

history = model.fit(x_train,
                    labels_train,
                    batch_size=32,
                    shuffle=True,
                    # important since we loaded cats first, dogs second.
                    epochs=10,
                    validation_data=(x_valid, labels_valid))


'''Step 3 Train the CNN
. One single convolution layer with 24 kernels,
· Two fully connected layers,
· Max pooling,
· Measure the Pearson correlation between predictions and validation labels.'''

fc_layer_size = 128
img_size = IMG_SIZE

conv_inputs = keras.Input(shape=(img_size[1], img_size[0], 3), name='ani_image')
conv_layer = layers.Conv2D(24, kernel_size=3, activation='relu')(conv_inputs)
conv_layer = layers.MaxPool2D(pool_size=(2, 2))(conv_layer)
conv_x = layers.Flatten(name='flattened_features')(conv_layer)  # turn image to vector.

conv_x = layers.Dense(fc_layer_size, activation='relu', name='first_layer')(conv_x)
conv_x = layers.Dense(fc_layer_size, activation='relu', name='second_layer')(conv_x)
conv_outputs = layers.Dense(1, activation='sigmoid', name='class')(conv_x)

conv_model = keras.Model(inputs=conv_inputs, outputs=conv_outputs)
# Original sample code used learning rate 1e-6 which is very low and caused no meaningful learning
customAdam = keras.optimizers.Adam(learning_rate=1e-4)
conv_model.compile(optimizer=customAdam,  # Optimizer
                   # Loss function to minimize
                   loss="binary_crossentropy",
                   # List of metrics to monitor - added accuracy for step 4
                   metrics=["binary_crossentropy", "mean_squared_error", "accuracy"])

# Epoch 5/5 loss: 1.6900 val_loss: 2.0413 val_mean_squared_error: 0.3688
print('# Fit model on training data step 3: single conv layer CNN')

history = conv_model.fit(x_train,
                         labels_train,  # we pass it th labels
                         # If the model is taking too long to train, make this bigger
                         # If it is taking too long to load for the first epoch, make this smaller
                         batch_size=32,
                         shuffle=True,
                         epochs=5,
                         # We pass it validation data to
                         # monitor loss and metrics
                         # at the end of each epoch
                         validation_data=(x_valid, labels_valid))

# Reshape preds to be 1D array for correlation calculation included in sample code
preds = conv_model.predict(x_valid).reshape(-1)
# also reshape labels to be 1D array for correlation calculation. Relevant again in step 4
labels_flat = labels_valid.reshape(-1)

corr = np.corrcoef(preds, labels_flat)[0][1]  # 0.15292172
print("Validation Pearson correlation: ", corr)

''' Step 4: Perform the following analysis:
1) Modify the model by adding another convolutional layer and use 48 kernels. What is the new correlation coefficient?
2) Assess the accuracy of the model.'''

conv2_inputs = keras.Input(shape=(img_size[1], img_size[0], 3), name='ani_image')

x = layers.Conv2D(24, kernel_size=3, activation='relu')(conv2_inputs)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(48, kernel_size=3, activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Flatten(name='flattened_features')(x)
x = layers.Dense(fc_layer_size, activation='relu', name='first_layer')(x)
x = layers.Dense(fc_layer_size, activation='relu', name='second_layer')(x)
conv2_outputs = layers.Dense(1, activation='sigmoid', name='class')(x)

conv_model_big = keras.Model(inputs=conv2_inputs, outputs=conv2_outputs)

customAdam = keras.optimizers.Adam(learning_rate=1e-4)
conv_model_big.compile(
    optimizer=customAdam,
    loss="binary_crossentropy",
    metrics=["accuracy", "mean_squared_error", "binary_crossentropy"]
)

cat_quantity = labels_flat.sum()

print('# Fit 2-conv CNN model on training data')
history_big = conv_model_big.fit(
    x_train,
    labels_train,
    batch_size=32,
    shuffle=True,
    epochs=10,
    validation_data=(x_valid, labels_valid)
)

preds_big = conv_model_big.predict(x_valid).reshape(-1)
corr_big = np.corrcoef(preds_big, labels_flat)[0][1]
pred_labels_big = (preds_big >= 0.5).astype("float32")
acc_big = (pred_labels_big == labels_flat).mean()

print("Step 4.1 - New validation Pearson correlation:", corr_big)
print("Step 4.2 - New validation accuracy @ 0.5 threshold:", acc_big)

# Step 5 : Visualize results and threshold analysis
sns.scatterplot(x=preds_big, y=labels_flat)
plt.title("Validation labels vs predicted probability (cat=1)")
plt.xlabel("Predicted probability of cat")
plt.ylabel("True label (1=cat, 0=dog)")
plt.show()

# Threshold analysis similar to starter code, but corrected
for i in range(1, 10):
    t = 0.1 * i
    selected = preds_big >= t
    if selected.sum() == 0:
        print("threshold:", t, "no predictions above threshold")
        continue
    frac_actually_cat = labels_flat[selected].mean()
    print("threshold:", t, "fraction actually cat among predicted cat:", frac_actually_cat)


def animal_pic(index):
    return Image.fromarray(x_valid[index])
# changed from x_valid[124] in sample code to xvalid[index] as this makes more sense


def cat_index(model_to_use, index):
    return float(model_to_use.predict(np.asarray([x_valid[index]]), verbose=0)[0][0])


# Step 6: Save the model
conv_model_big.save("conv_model_big.h5")

# Step 7: Create an interface to interact with the model
def show_prediction(index, model_to_use):
    """
    Prints the probability of being a cat and shows the image.
    """
    if index < 0 or index >= len(x_valid):
        print("Index out of range. Must be between 0 and {}.".format(len(x_valid) - 1))
        return

    # Model expects a batch dimension: (1, H, W, 3)
    prob_cat = float(model_to_use.predict(np.asarray([x_valid[index]]), verbose=0)[0][0])

    print("Index:", index)
    print("Probability of being a cat:", prob_cat)

    img = Image.fromarray(x_valid[index].astype("uint8"))
    img.show()

# CLI loop for model interface
def run_model_interface(model_to_use):
    """
    CLI loop that asks the user for an index and shows prediction + image.
    Type 'q' to quit.
    """
    print("\nStep 7 Model Interface")
    print("Type an index from 0 to {} or 'q' to quit.\n".format(len(x_valid) - 1))

    while True:
        user_in = input("Enter image index (or 'q'): ").strip()

        if user_in.lower() == "q":
            print("Exiting.")
            break

        try:
            idx = int(user_in)
        except ValueError:
            print("Please enter a valid integer index.")
            continue

        show_prediction(idx, model_to_use)

# Main program loop to choose model and run interface
def main():
    while True:
        print("\nChoose which model to use for predictions:")
        print("1: Single Conv Layer Model (in-memory)")
        print("2: Two Conv Layer Model (load from disk: conv_model_big.h5)")
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            model_to_use = conv_model
            print("Using Single Conv Layer Model (in-memory).")

        elif choice == "2":
            try:
                model_to_use = tf.keras.models.load_model("conv_model_big.h5")
                print("Loaded Two Conv Layer Model from disk: conv_model_big.h5")
            except Exception as e:
                print("Could not load saved model 'conv_model_big.h5'.")
                print("Error:", e)
                print("Falling back to in-memory conv_model_big.")
                model_to_use = conv_model_big

        else:
            print("Invalid choice. Please enter 1 or 2.")
            continue

        run_model_interface(model_to_use)

        print("\nTry another model?")
        again = input("Enter 'y' to try again, any other key to exit: ").strip().lower()
        if again != "y":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()