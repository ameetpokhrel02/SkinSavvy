#!/usr/bin/env python3

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print("TensorFlow imported successfully!")
except ImportError as e:
    print(f"TensorFlow import error: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import error: {e}")

try:
    from PIL import Image
    print("PIL/Pillow imported successfully!")
except ImportError as e:
    print(f"PIL import error: {e}")
