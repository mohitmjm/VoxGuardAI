"""Serve the existing FastAPI endpoints and a small Gradio status page."""
import os
import gradio as gr
import uvicorn

# Set before importing backend.app, where the CORS middleware is configured.
os.environ.setdefault('VOXGUARD_CORS_ORIGINS', 'https://voxguardai-frontend.vercel.app')
os.environ.setdefault('VOXGUARD_DEVICE', 'cpu')

from backend.app import app as api
from ml.download_model import main as download_checkpoint

with gr.Blocks() as status_page:
    gr.Markdown('''# VoxGuard AI inference service
Use the application at https://voxguardai-frontend.vercel.app.

API status: [/api/health](/api/health) · Model: [/api/model/info](/api/model/info)

Real pretrained AASIST inference. Scores are uncalibrated; default decisions remain Uncertain.
''')

# Existing API routes precede the root mount, preserving /api/analyze and /docs.
app = gr.mount_gradio_app(api, status_page, path='/')

if __name__ == '__main__':
    download_checkpoint()
    uvicorn.run(app, host='0.0.0.0', port=7860, workers=1)
