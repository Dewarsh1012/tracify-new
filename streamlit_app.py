"""
Root Streamlit entrypoint for Streamlit Cloud Deployment.
Redirects to project/streamlit_app.py with proper path resolution.
"""

import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(root_dir, "project")

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

os.chdir(project_dir)

app_path = os.path.join(project_dir, "streamlit_app.py")
with open(app_path, "r") as f:
    code = compile(f.read(), app_path, "exec")
    exec(code, {"__file__": app_path, "__name__": "__main__"})
