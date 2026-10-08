from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from .config import RunConfig
from .privacy import calibrate, precision_diag
from .data import dataset_root, resolve_device


def main() -> None:
    p=argparse.ArgumentParser(description="Exact linear pixel-gradient reconstruction probe from the source notebook.")
    p.add_argument("--output", default="results/experiments/pixel_probe")
    args=p.parse_args()
    cfg=RunConfig()
    device=resolve_device(cfg.device)
    raw_test=datasets.CIFAR10(str(dataset_root(cfg)),train=False,download=True)
    attack_ids=[0,1,2]
    probe_cal=calibrate(4.0,cfg.delta,1,2*cfg.clip,cfg.num_clients,cfg.max_colluders,cfg.max_stragglers)
    H=cfg.num_clients-cfg.max_colluders
    probe_light_var=1/precision_diag(probe_cal.u,probe_cal.k,H,0)
    probe=nn.Linear(3072,10).to(device)
    rows=[];fig,axs=plt.subplots(len(attack_ids),5,figsize=(16,9))
    for row,idx in enumerate(attack_ids):
        x=transforms.ToTensor()(raw_test[idx][0]).to(device)
        label=torch.tensor([raw_test[idx][1]],device=device)
        gw,gb=torch.autograd.grad(F.cross_entropy(probe(x.flatten()[None]),label),tuple(probe.parameters()))
        scale=(cfg.clip/torch.sqrt(gw.square().sum()+gb.square().sum())).clamp(max=1)
        clean=torch.cat([(gw*scale).flatten(),gb*scale])
        original=x.permute(1,2,0).cpu().numpy()
        axs[row,0].imshow(original);axs[row,0].set_title(raw_test.classes[label.item()])
        methods=[("No DP",0.0),("Vanilla DP",probe_cal.normal_var),("SMPC+DP",probe_cal.normal_var),("LightDP",probe_light_var)]
        for col,(name,var) in enumerate(methods,1):
            obs=clean+math.sqrt(var)*torch.randn_like(clean)
            w=obs[:30720].view(10,3072);b=obs[30720:]
            j=b.abs().argmax();den=b[j].clamp(min=1e-12) if b[j]>=0 else b[j].clamp(max=-1e-12)
            recovered=(w[j]/den).view(3,32,32).clamp(0,1).permute(1,2,0).cpu().numpy()
            mse=float(np.mean((original-recovered)**2));ssim=float(structural_similarity(original,recovered,channel_axis=2,data_range=1))
            rows.append(dict(image_id=idx,method=name,mse=mse,psnr=-10*np.log10(max(mse,1e-15)),ssim=ssim))
            axs[row,col].imshow(recovered);axs[row,col].set_title(f"{name} | SSIM {ssim:.3f}")
    for ax in axs.flat:ax.axis("off")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(out/"00_exact_pixel_reconstruction.png",dpi=180);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/"pixel_attack.csv",index=False)
    print(out.resolve())

if __name__ == "__main__": main()
