"""ZeroGPU-backed AASIST with the existing Vercel-facing FastAPI contract."""
import os
from contextlib import asynccontextmanager

# Import spaces before torch so ZeroGPU can install its CUDA hooks.
import spaces
import gradio as gr
from starlette.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

os.environ.setdefault('VOXGUARD_CORS_ORIGINS', 'https://voxguardai-frontend.vercel.app')
os.environ['VOXGUARD_DEVICE'] = 'cuda'

import backend.app as backend
from ml.download_model import main as download_checkpoint
from ml.service import ModelService

_worker_model = None

@spaces.GPU(duration=60)
def gpu_analyze(samples, sample_rate=16000):
    """Run the actual detector on an allocated GPU, never a dummy GPU task."""
    global _worker_model
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('ZeroGPU did not allocate a CUDA device.')
    if _worker_model is None:
        _worker_model = ModelService()
    return _worker_model.analyze(samples, sample_rate=sample_rate)

class ZeroGPUModel:
    def __init__(self):
        # Metadata only: no CUDA model is loaded in the web-server process.
        self.metadata = ModelService()

    def info(self):
        result = self.metadata.info()
        result['execution'] = 'Hugging Face ZeroGPU; allocation and quotas apply'
        return result

    def analyze(self, samples, sample_rate=16000):
        return gpu_analyze(samples, sample_rate)

@asynccontextmanager
async def lifespan(app):
    # Register the existing upload/health routes before accepting requests.
    app.router.routes.extend(
        route for route in backend.app.router.routes
        if getattr(route, 'path', '').startswith('/api/')
    )
    yield

with gr.Blocks() as status_page:
    gr.Markdown('''# VoxGuard AI inference service
Use https://voxguardai-frontend.vercel.app for audio uploads.

[API health](/api/health) · [Model information](/api/model/info)

Official pretrained AASIST runs inside an allocated ZeroGPU task.
Scores are uncalibrated; default decisions remain Uncertain.
Free-host GPU quotas and queue delays apply.
''')

if __name__ == '__main__':
    download_checkpoint()
    backend._model = ZeroGPUModel()
    origins = [v.strip() for v in os.environ['VOXGUARD_CORS_ORIGINS'].split(',') if v.strip()]
    # spaces wraps launch() to register tasks and initialize the GPU lifecycle.
    status_page.launch(
        server_name='0.0.0.0', server_port=7860,
        app_kwargs={
            'lifespan': lifespan,
            'middleware': [
                Middleware(CORSMiddleware, allow_origins=origins,
                           allow_methods=['GET', 'POST'], allow_headers=['Content-Type']),
                Middleware(backend.BodyLimitMiddleware),
            ],
        },
    )
