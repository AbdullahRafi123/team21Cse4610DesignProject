# Research standards for this repository

These standards govern new experiments and any future paper or report based on this code. They draw on the [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist) and the [FAIR principles for research data and software](https://www.nature.com/articles/sdata201618). The target venue's current policies always take precedence.

## 1. Define the study before running it

For every confirmatory experiment, create a protocol from [`EXPERIMENT_PROTOCOL_TEMPLATE.md`](EXPERIMENT_PROTOCOL_TEMPLATE.md) before inspecting the final outcomes. Record:

- The research question, primary hypothesis, and primary outcome metric.
- The model, data version, split, preprocessing, client partitioning, and evaluation procedure.
- Each baseline, its configuration, and any tuning budget. Keep data, model capacity, training budget, and compute conditions comparable.
- The number of independent seeds/runs and the planned statistical analysis. Justify deviations from the protocol.
- The threat model and assumptions for privacy or security claims.

Use smoke runs only to verify software operation. Label exploratory sweeps as exploratory. Do not select only favorable seeds, metrics, or parameter settings for reporting.

## 2. Make results reproducible

Each run must retain its exact effective configuration and provenance in `results/experiments/<run-id>/`. Keep the run tag unique and record:

- Git commit/revision, code changes, Python and dependency versions, operating system, hardware, and accelerator details when relevant.
- Dataset source/version, license, split and partition procedure, preprocessing, and any downloaded or pretrained assets and their version.
- All hyperparameters, random seeds, client count, rounds, privacy parameters, and deviations from defaults.
- Per-round evaluation metrics, training/aggregation metrics, failures, and the complete run log.

Use the provided configuration and scripts as the canonical path. Do not edit generated CSV/JSON logs by hand. Store large or restricted artifacts outside Git and document how to obtain them. Keep small logs and summaries in a reviewable, machine-readable format.

## 3. Analyze and report honestly

- Report every planned baseline and outcome, including null, negative, and failed results. Distinguish protocol deviations and incomplete runs.
- For confirmatory claims, run independent seeds and report the number of runs, aggregate statistic, uncertainty interval, and statistical procedure. Use paired analysis when runs share controlled randomness. Explain multiple comparisons and any exclusions.
- Include absolute values and units, not only relative improvements. State hardware, wall-clock timing boundaries, memory measurement method, and whether preprocessing/cache time is included.
- Separate directly measured quantities from projections, estimates, and theoretical bounds. State assumptions and show the calculation for each estimate.
- Do not infer cryptographic security from a simulation. State the adversary, trust assumptions, threat model, and what the implementation does not model.
- Ensure every table and figure can be regenerated from versioned code and retained logs. Include the command, config, run IDs, and script that produced it.

## 4. Document data, models, and limits

The active project uses CIFAR-10 and can use torchvision's pretrained ResNet-18 weights. Cite the [CIFAR-10 dataset and its recommended technical report](https://www.cs.toronto.edu/~kriz/cifar.html) and the exact [torchvision ResNet-18 weight variant](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) used. Record the license/terms and any preprocessing. A dependency or pretrained checkpoint is not automatically redistributable just because it downloads successfully.

Preserve the current scope statements: `lightdp` simulates pairwise masks with matching seeds; it is not a key-exchange implementation. `smpc_dp` is an ideal secure-aggregation baseline, not a cryptographic transport implementation. Reconstruction conclusions apply only to the documented attack, inputs, and threat model. Do not generalize beyond measured evidence.

Before release, document intended use, limitations, known failure modes, and any privacy or societal risks relevant to the claims. Do not publish personal or restricted data, secrets, or unreviewed model artifacts.

## 5. Cite and release responsibly

- Cite prior work, datasets, model weights, libraries, and any adapted implementation where they support the method or claims.
- Cite the framework paper when describing Flower: Beutel et al., [Flower: A Friendly Federated Learning Research Framework](https://arxiv.org/abs/2007.14390). Cite the model paper when describing ResNet: He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). Follow the CIFAR-10 page's request to cite Krizhevsky's technical report.
- Keep the software citation metadata in `CITATION.cff` accurate. Confirm the full author list and contribution order with the team before a paper or archival release; repository ownership is not an authorship test.
- Follow the submission venue's current reproducibility, ethics, anonymization, artifact, and disclosure requirements. The NeurIPS checklist is a useful example, not a substitute for the chosen venue's policy.
- Give released artifacts a version or immutable commit identifier and instructions that start from a clean clone. Report any dependency, data, or hardware requirement that prevents exact reproduction.

## 6. Release readiness gate

Do not label an experiment or paper artifact publication-ready until all applicable items are complete:

- The protocol is finalized, the complete planned runs are present, and conclusions match the protocol and uncertainty in the data.
- A clean environment can install the project and reproduce the reported tables/figures from retained logs using documented commands.
- The release has a pinned code revision, dependency versions, data/model provenance, license/attribution review, and complete author/contribution confirmation.
- The paper reports limitations, threat-model assumptions, computational costs, and negative results, and has completed the target venue's current checklist.
- A second team member has reviewed the experiment protocol, result scripts, and claims against the generated artifacts.

## Project status note

Archived notebook results are historical records. A successful smoke run demonstrates that the application starts and writes logs; it does not reproduce the paper-style findings. The existing measured report must not be presented as a current independent replication without rerunning its complete protocol and preserving the resulting provenance.
