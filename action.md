# WDMF-Net Server Actions

Server checkout: `/home/xxj/code/DINOCLIP_comparative_test/WDMF-Net`

## Pull updates

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/WDMF-Net
git switch main
git pull --ff-only origin main
```

## Create the profiling environment

The upstream `requirements.txt` pins PyTorch 2.1.0, which predates RTX 5090 support. Use the server setup script instead; it installs PyTorch 2.7.1 with CUDA 12.8 in an isolated Python 3.10 environment:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/WDMF-Net
bash setup_server_env.sh
conda activate wdmf-rs
```

## Parameters and FLOPs

WDMF-Net's README specifies 256x256 input patches. The profiler uses batch 1, a paired RGB input, and random initialization; ImageNet MobileNet weights are not needed to measure parameter/FLOP counts. `fvcore` unsupported operators are listed in the JSON, and those operations are excluded from its FLOP total.

```bash
CUDA_VISIBLE_DEVICES=0 python profile_model.py --device cuda:0 \
  --output WDMF-Net_work_dirs/profile_256.json
```

Read `Size(M)` and `Flops(G)` from stdout; the JSON also records the fvcore convention, unsupported operators, and any tracing error.
