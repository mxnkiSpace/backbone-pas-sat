# Inferencia de backbone con la GNN de NeuroBack.
#
# Adaptado de NeuroBack (predict.py: predict_single + merge_wcc_preds).
#
# Origen: NeuroBack (Wang et al., ICLR 2024). MIT License,
# Copyright (c) 2024 wenxiwang. Ver LICENSES/NEUROBACK-LICENSE.


import argparse
import os

import torch

from model.gt_model import GTModel
from preprocessing.cnf_to_graph import cnf_to_bipartite

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "model", "finetune-best.ptg.zip")


def load_model(model_path=DEFAULT_CKPT, device="cpu"):
    """Instancia GTModel(3, 3) y carga el checkpoint fine-tuneado."""
    model = GTModel(3, 3)
    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def _predict_data(data, model, device):
    """Corre el modelo sobre un objeto Data (un WCC) -> {variable: score}.

    Replica la transformación de NeuroBack: aristas dirigidas -> bidireccionales
    y casteo a float antes del forward.
    """
    reverse = data.edge_index.index_select(0, torch.LongTensor([1, 0]))
    edge_index = torch.cat([data.edge_index, reverse], dim=1).long()
    edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0).float()
    x = data.x.float()

    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    with torch.no_grad():
        pred = model(x, edge_index, edge_attr)

    n2v = data.n2v.cpu().numpy().tolist()
    pred = pred.cpu()

    scores = {}
    for n, v in enumerate(n2v):
        scores[int(v)] = float(pred[n].item())
    return scores


def predict_backbone(cnf_path, model=None, model_path=DEFAULT_CKPT, device="cpu"):
    """CNF -> {variable: score de backbone}.

    El grafo se parte en WCC; cada variable vive en exactamente una WCC, así
    que la unión de scores cubre todas las variables predichas.
    """
    if model is None:
        model = load_model(model_path, device)

    data_lst = cnf_to_bipartite(cnf_path)
    if not data_lst:
        return {}

    scores = {}
    for data in data_lst:
        scores.update(_predict_data(data, model, device))
    return scores


def main():
    parser = argparse.ArgumentParser(description="Predice scores de backbone para un CNF.")
    parser.add_argument("cnf", help="Ruta al CNF (.cnf, .cnf.xz, .gz, .bz2, ...)")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Ruta al checkpoint .ptg.zip")
    parser.add_argument("--cuda", action="store_true", help="Usar GPU si está disponible")
    parser.add_argument("-o", "--out", help="Archivo de salida 'variable,score' (default: stdout)")
    parser.add_argument("--top", type=int, default=None, help="Mostrar solo las N variables con mayor score")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    scores = predict_backbone(args.cnf, model_path=args.ckpt, device=device)

    rows = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if args.top is not None:
        rows = rows[: args.top]

    lines = [f"{v},{s:.6f}" for v, s in rows]
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{len(rows)} variables -> {args.out}")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
