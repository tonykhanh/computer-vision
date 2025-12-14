import os

# Model Configuration
MODEL_PATH = "mobilenet/mobilenet.h5"
INPUT_SHAPE = (128, 128)

# Categories
CATEGORIES = [
    'annona', 'apples', 'bananas', 'lemons', 
    'mango', 'oranges', 'tomatoes', 'human', 
    'pen', 'phone'
]

# UI Configuration
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)

# Thresholds
PREDICTION_THRESHOLD = 0.5 # Not strictly used in original, but good practice
