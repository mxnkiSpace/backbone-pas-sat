# backbone-pas-sat
Extending Predict-and-Search to pure SAT solving via backbone-guided trust regions, built on a NeuroBack-trained GNN.

### About Benchmark instances
Benchmark instances come from two complementary sources, chosen to test the trust region under both controlled and real-world conditions. The satlib/cbs_backbone_controlled set (SATLIB, Controlled Backbone Size) provides random 3-SAT instances with the backbone proportion fixed by construction (10–90%), allowing direct measurement of how trust region effectiveness scales with backbone density without a separate extraction step. The satlib/random_3sat, graph_coloring, and planning sets add domain diversity for contrast. Finally, satcomp_holdout/{2024,2025} (SAT Competition main track, via the Global Benchmark Database) provides large, real-world industrial/application instances that were not part of NeuroBack's pre-training, fine-tuning, or original evaluation — ensuring a leakage-free test of the trained backbone model.

## Attribution & third-party code

This project builds on **NeuroBack** (Wang et al., ICLR 2024). Two components are
ported from the NeuroBack codebase and are **not** original contributions of this thesis:

- [model/gt_model.py](model/gt_model.py) — the Bipartite Graph Transformer GNN architecture, copied
  verbatim so it can load the trained checkpoint. Its exact shape defines the weights.
- [preprocessing/cnf_to_graph.py](preprocessing/cnf_to_graph.py) — the CNF → bipartite-graph encoding.
  Adapted for inference-only use (see the file header for the precise diff); the
  graph *format* is identical to what NeuroBack saw during training.

NeuroBack is distributed under the **MIT License** (Copyright © 2024 wenxiwang).
The full upstream license is reproduced in [LICENSES/NEUROBACK-LICENSE](LICENSES/NEUROBACK-LICENSE),
as the MIT terms require for substantial portions of the software.

The novel contribution of this thesis — the backbone-guided trust region within the
Predict-and-Search paradigm — lives in `trust_region/`.

If you use this work, please cite the NeuroBack paper:

```bibtex
@inproceedings{wang2024neuroback,
  title     = {NeuroBack: Improving {CDCL} {SAT} Solving using Graph Neural Networks},
  author    = {Wenxi Wang and Yang Hu and Mohit Tiwari and Sarfraz Khurshid and Kenneth McMillan and Risto Miikkulainen},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024},
}
```

### Model checkpoint provenance

The versioned checkpoint `model/finetune-best.ptg.zip` (1.5 MB) is a backbone-prediction
model **trained from scratch for this thesis**, using the NeuroBack GNN architecture
(MIT-licensed). The pre-trained weights released by the NeuroBack authors were *not* used;
only their architecture and graph encoding were reused. The weights in this checkpoint are
therefore an original artifact of this work. It is committed directly to keep the repository
self-contained and reproducible.

The training/fine-tuning data was the **DataBack** dataset published by the NeuroBack authors
on [HuggingFace](https://huggingface.co/datasets/neuroback/DataBack); please refer to that
dataset's own terms of use.
