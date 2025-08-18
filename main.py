"""📦 Import Libraries"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import vgg16
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

"""🗂️ Data Preparation"""

# Define dataset paths
train_path = "dataset/train"
valid_path = "dataset/valid"
test_path = "dataset/test"

# Load and preprocess training data
train_batches = ImageDataGenerator(
    preprocessing_function=vgg16.preprocess_input
).flow_from_directory(train_path, target_size=(224, 224), batch_size=10)

# Load and preprocess validation data
valid_batches = ImageDataGenerator(
    preprocessing_function=vgg16.preprocess_input
).flow_from_directory(valid_path, target_size=(224, 224), batch_size=30)

# Load and preprocess test data (no shuffle to preserve order)
test_batches = ImageDataGenerator(
    preprocessing_function=vgg16.preprocess_input
).flow_from_directory(test_path, target_size=(224, 224), batch_size=50, shuffle=False)

"""🧠 Download VGG16 & Create Custom Network"""

# Load pretrained VGG16 model without top layers
base_model = vgg16.VGG16(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
)

# Freeze all layers except the last 5 for fine-tuning
for layer in base_model.layers[:-5]:
    layer.trainable = False
base_model.summary()

"""🏗️ Build Full Model (+ Classifier)"""

# Save output of last layer of VGG as input for custom classifier
last_layer_output = base_model.output

# Add classification layer (10 classes, softmax activation)
x = Dense(units=10, activation="softmax", name="softmax")(last_layer_output)

# Define new full model (base + classifier)
new_model = Model(inputs=base_model.input, outputs=x, name="Full_Model")

new_model.summary()

"""⚙️ Compile the Model"""

# Compile with Adam optimizer and categorical crossentropy loss
new_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

"""📈 Train the Model"""

# Save only the best model during training
checkpoint_path = "model.sign_language.keras"
checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)

# Train model with training + validation data
history = new_model.fit(
    train_batches,
    steps_per_epoch=18,
    validation_data=valid_batches,
    validation_steps=3,
    epochs=20,
    verbose=1,
    callbacks=[checkpointer],
)

"""💾 Load the Best (Saved) Model"""

# Load weights of the best saved checkpoint
new_model.load_weights("model.sign_language.keras")

"""🧪 Evaluate the Model"""

# Evaluate model on test dataset
score = new_model.evaluate(test_batches)
print("\nTest loss:", score[0])
print("\nTest accuracy:", score[1])

"""🔮 Visualize Predictions"""

# Get predictions on the test set
y_hat = new_model.predict(test_batches)

# Labels for 10 sign language classes (digits 0–9)
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

num_test_images = test_batches.samples

# Plot 32 random test images with predictions vs ground truth
fig = plt.figure(figsize=(20, 8))
for i, idx in enumerate(np.random.choice(num_test_images, size=32, replace=False)):
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])

    # Load and display test image
    img_path = test_batches.filepaths[idx]
    img = load_img(img_path, target_size=(224, 224))
    ax.imshow(img)

    # Predicted vs true class index
    pred_idx = np.argmax(y_hat[idx])
    true_idx = test_batches.classes[idx]
    ax.set_title(
        f"{labels[pred_idx]} ({labels[true_idx]})",
        color=("green" if pred_idx == true_idx else "red"),
    )

"""📊 Confusion Matrix"""

# Get true and predicted labels
y_true = test_batches.classes
y_pred = np.argmax(y_hat, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
plt.title("Confusion Matrix")
plt.show()
