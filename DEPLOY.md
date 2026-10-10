# Deploying the demo to Hugging Face Spaces

Nothing here creates a Space for you. The repo ships what a Docker Space needs: `Dockerfile`
(builds the React dashboard, installs core deps only, no torch, runs the API in `DEMO_MODE=1` on
port 7860) and a Space card (`deploy/huggingface/README.md`).

The Space serves the **synthetic demo only**. Do not upload real CCTV footage or student data to a public Space.

## Test the image locally

```bash
docker build -t campus-safety-demo .
docker run --rm -p 7860:7860 campus-safety-demo   # open http://localhost:7860
```

## Steps

1. Create a Space at https://huggingface.co/new-space: SDK **Docker**, hardware **CPU basic** (free).
2. Push the repo to it, with the Space card as the root README (Spaces read config from its front matter):

   ```bash
   git clone https://huggingface.co/spaces/<user>/<space> space
   git archive HEAD | tar -x -C space
   cp deploy/huggingface/README.md space/README.md
   rm -f space/demo.gif   # legacy README screenshot of real campus footage; keep it off the Space
   cd space
   git lfs install && git lfs track "*.jpg" "*.gif" "*.npz"   # the Hub rejects binaries outside LFS/Xet
   git add -A && git commit -m "Deploy" && git push
   ```

## Settings

No secrets are needed. Do **not** set `TELEGRAM_TOKEN` or `DEMO_MODE=0` on a public Space:
demo mode is what keeps alerts mocked.
