"""
Root Streamlit entrypoint for Streamlit Cloud Deployment.
Redirects to project/streamlit_app.py while setting up paths.
"""

import os
import sys

# Add project directory to Python sys.path
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project")
sys.path.insert(0, project_dir)
os.chdir(project_dir)

# Run the Streamlit app
with open("streamlit_app.py") as f:
    exec(f.read())
