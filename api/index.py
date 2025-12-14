import os
import sys

# Add the parent directory to sys.path to allow importing app.py from root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app import app as application

# Vercel expects a variable named 'app' or 'handler' or looks for WSGI app.
# We aliased it to application, but verify what Vercel needs.
# Actually, simply exposing 'app' is usually enough if it's a Flask object.
app = application
