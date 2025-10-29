import tensorflow as tf
from tensorflow.keras.datasets import mnist

print("="*60)
print("🔍 TensorFlow Environment Check")
print("="*60)

# TensorFlow & Keras versions
print(f"TensorFlow version: {tf.__version__}")
try:
    import keras
    print(f"Keras version: {keras.__version__}")
except Exception:
    print("Keras version: (included inside TensorFlow)")

print("\n🧠 Devices found:")
for device in tf.config.list_physical_devices():
    print(" -", device)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("\n✅ GPU detected! TensorFlow is using Metal (Apple GPU).")
else:
    print("\n⚠️ No GPU detected. Using CPU only.")
    print("   If you have an M1/M2 Mac, install the Metal plugin:")
    print("   pip install tensorflow-metal")

# Small computation test
print("\n🧪 Running a small TensorFlow test computation...")
a = tf.random.normal((1000, 1000))
b = tf.random.normal((1000, 1000))
c = tf.matmul(a, b)
print("Matrix multiply completed successfully. ✅")

# MNIST data test
print("\n📦 Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"MNIST shapes -> x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
print("Sample pixel value range:", x_train.min(), "to", x_train.max())

print("\n🎉 Environment looks good! You’re ready to train your CNN.")
print("="*60)
