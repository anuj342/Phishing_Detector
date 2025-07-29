try:
    # Import libraries
    import tensorflow as tf
    import keras
    import cv2 as cv

    # Print version numbers
    print(f"✅ TensorFlow version: {tf.__version__}")
    print(f"✅ Keras version: {keras.__version__}")
    print(f"✅ OpenCV version: {cv.__version__}")
    print("\nAll libraries were imported successfully!")

except ImportError as e:
    print(f"❌ Error importing libraries: {e}")
    print("Please check your installation.")