# Deployment without Docker

Keep both components in the same GitHub repository. Deploy the frontend and backend as separate services. Closing your laptop does not stop services hosted by a provider. Continuous availability requires a backend plan that does not sleep on idle; free tiers may suspend or restrict inference. No hosting plan or cost is assumed here.

## Backend: ordinary Python web service

Choose a provider that supports a persistent Python process, PyTorch CPU wheels, installation of FFmpeg **and** FFprobe, writable temporary storage, HTTPS, and downloading the official model checkpoint. A generic short-lived function host is unsuitable without separately verifying its dependency size, execution duration, and memory limits. Vercel is used for the frontend in this configuration.

Set the repository root as the working directory, and Python 3.11+ (3.12 was locally tested). Install FFmpeg with the host's supported system-package mechanism. If it cannot provide both binaries, select another host. Docker is optional.

Build commands:

```sh
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend/requirements.txt -r ml/requirements.txt
python -m ml.download_model
python -m pip check
```

Start command (POSIX host; replace the port if the provider uses another setting):

```sh
python -m uvicorn backend.app:app --host 0.0.0.0 --port "$PORT" --workers 1
```

One worker avoids loading multiple copies of the model; the service allows one analysis at a time and returns HTTP 429 when occupied. Set the provider health path to `/api/health`. That endpoint reports software availability; verify real inference separately after deployment. Monitor memory, CPU and latency with the maximum supported recording before choosing a plan. Cloud memory requirements have not been benchmarked. Provide headroom for PyTorch, FFmpeg, the model and bounded upload buffers. Do not assume a free-tier instance is sufficient.

Environment settings:

| Name | Example / meaning |
|---|---|
| `VOXGUARD_DEVICE` | `cpu` |
| `VOXGUARD_TORCH_THREADS` | `2`, tune for the allocated CPU |
| `VOXGUARD_CORS_ORIGINS` | Exact frontend HTTPS origin, e.g. `https://your-project.vercel.app`; comma-separated for multiple origins |
| `VOXGUARD_MAX_UPLOAD_MB` | `20` by default |
| `VOXGUARD_MAX_DURATION_SECONDS` | `120` by default |

No API key is required. Keep CORS origins explicit. Preview deployments have different origins; add only the origins you intend to support. Public inference is unauthenticated: put provider-level request-rate and resource limits in place before allowing unrestricted traffic. Uploaded recordings must not be logged or retained; do not configure body capture in monitoring systems. Prefer a provider-managed HTTPS endpoint and reverse proxy with request-size and timeout settings compatible with the application.

## Frontend: Vercel

Import the GitHub repository and select the branch containing the application.

| Setting | Value |
|---|---|
| Root directory | `frontend` |
| Framework | Vite |
| Build command | `npm run build` |
| Output directory | `dist` |
| Install command | `npm ci` |
| Node.js | 22+ |
| `VITE_API_BASE_URL` | Public HTTPS backend origin without a trailing slash, e.g. `https://your-backend.example` |

This variable is embedded at build time. Redeploy the frontend after changing it. The Vite `/api` proxy is for development only; it is not part of static Vercel hosting. Configure the backend CORS origin using the final frontend URL.

## Release verification

1. Run `bash scripts/verify.sh` locally before release; CI also runs backend and frontend checks.
2. Confirm the hosted `/api/health` and `/api/model/info` endpoints respond over HTTPS and FFmpeg is available.
3. Upload a consented short speech recording using the deployed frontend. Verify real inference returns a finite score and segment information. Check meaningful errors for invalid uploads.
4. Verify the recording is not retained in application storage or logs. Test from a second device with the development laptop turned off.
5. Confirm the provider's no-sleep policy, restart behavior and resource limits. Check inference after a backend restart; live processes and in-memory model state do not survive restarts.

Do not claim benchmark accuracy from deployment smoke tests. Without validated thresholds, Uncertain is the intended scientific result; real model scores are still returned. Hindi, Marathi and modern generator performance remain unevaluated. This document supplies instructions only: it does not deploy a service or establish public availability.
