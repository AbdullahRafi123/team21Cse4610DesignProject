from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path("results")
rows=[];histories=[]
for folder in sorted(ROOT.glob("*")):
    hp=folder/"history.csv";cp=folder/"run_config.json"
    if not hp.exists() or not cp.exists(): continue
    hist=pd.read_csv(hp);cfg=json.loads(cp.read_text())
    hist["run"]=folder.name;histories.append(hist)
    last=hist.iloc[-1].to_dict();last.update({"run":folder.name,"method":cfg["method"],"N":cfg["num_clients"],"Cmax":cfg["max_colluders"],"Smax":cfg["max_stragglers"],"clip":cfg["clip"],"lr":cfg["learning_rate"]})
    rows.append(last)
if not rows:
    print("No finished results found under results/");raise SystemExit(0)
out=ROOT/"summary";out.mkdir(exist_ok=True)
final=pd.DataFrame(rows).sort_values("run");final.to_csv(out/"all_final_results.csv",index=False)
all_hist=pd.concat(histories,ignore_index=True);all_hist.to_csv(out/"history.csv",index=False)

main=final[final.run.str.match(r"^(No_DP|Vanilla_local_noise_adding_eps|SMPC\+DP_eps|LightDP_eps)")]
if not main.empty:
    fig,ax=plt.subplots(figsize=(10,5))
    for run in main.run:
        part=all_hist[all_hist.run==run];ax.plot(part["round"],part["accuracy"],marker="o",label=run)
    ax.set_xlabel("Communication round");ax.set_ylabel("Test accuracy (%)");ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/"main_accuracy.png",dpi=180);plt.close(fig)

private=final[final.run.str.match(r"^(Vanilla_local_noise_adding_eps|SMPC\+DP_eps|LightDP_eps)")]
if not private.empty:
    private=private.copy();private["epsilon_target"]=private.run.str.extract(r"eps(\d+)").astype(float)
    fig,ax=plt.subplots(figsize=(8,5))
    for prefix,label in [("LightDP_eps","LightDP"),("SMPC+DP_eps","SMPC+DP ideal"),("Vanilla_local_noise_adding_eps","Vanilla local DP")]:
        p=private[private.run.str.startswith(prefix)].sort_values("epsilon_target")
        if len(p): ax.plot(p.epsilon_target,p.accuracy,marker="o",label=label)
    ax.set_xlabel("Target epsilon");ax.set_ylabel("Final test accuracy (%)");ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/"epsilon_comparison.png",dpi=180);plt.close(fig)
print("Wrote",out.resolve())
