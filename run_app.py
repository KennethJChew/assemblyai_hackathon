import subprocess

STREAMLIT_APP_PATH = "stable_diffusion\optimizedSD\streamlit_app.py"
subprocess.run(f"streamlit run {STREAMLIT_APP_PATH}")