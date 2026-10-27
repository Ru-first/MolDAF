import argparse
import os
import time
import warnings

import random
import deepchem as dc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import MolDAF
from utils import compute_auc

from new_pubchemfp import GetPubChemFPs

from rdkit import RDLogger
# 关闭 RDKit warning
RDLogger.DisableLog('rdApp.warning')

warnings.filterwarnings("ignore")



classification_loaders = {
    "bace_classification": dc.molnet.load_bace_classification,
    "bbbp": dc.molnet.load_bbbp,
    "clintox": dc.molnet.load_clintox,
    "sider": dc.molnet.load_sider,
    "tox21": dc.molnet.load_tox21,
    "muv": dc.molnet.load_muv,
}


def log_message(message, log_fp=None):
    print(message)
    if log_fp is not None:
        print(message, file=log_fp)

def load_molformer_embeddings(output_dir_path, use_molformer, expected_sizes,):
    train_size, valid_size, test_size = expected_sizes
    if not use_molformer:
        return (
            np.zeros((train_size, 0), dtype=np.float32),
            np.zeros((valid_size, 0), dtype=np.float32),
            np.zeros((test_size, 0), dtype=np.float32),
        )

    path = os.path.join(output_dir_path, "Molformer_Emb_2025.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Molformer embeddings not found: {path}. "
            "Use --no-molformer to run without this modality."
        )

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "h5py is required when --use-molformer is enabled. "
            "Install h5py or run with --no-molformer."
        ) from exc

    with h5py.File(path, "r") as f:
        train_fp = f["train_fp"][:]
        valid_fp = f["valid_fp"][:]
        test_fp = f["test_fp"][:]

    if (len(train_fp), len(valid_fp), len(test_fp)) != expected_sizes:
        raise ValueError(
            "Molformer embedding size mismatch: "
            f"expected {expected_sizes}, "
            f"got {(len(train_fp), len(valid_fp), len(test_fp))}"
        )

    return train_fp.astype(np.float32), valid_fp.astype(np.float32), test_fp.astype(np.float32)


def extract_fp_and_smiles(dataset):
    """
    Generate three fingerprint views independently.

    Current dimensions:
      MACCS   = 167
      ErG     = 441
      PubChem = 881
      Total   = 1489
    """
    fingerprints = []

    for smi in dataset.ids:
        mol_obj = Chem.MolFromSmiles(smi)

        if mol_obj is None:
            fingerprints.append({
                "maccs": np.zeros(167, dtype=np.float32),
                "erg": np.zeros(441, dtype=np.float32),
                "pubchem": np.zeros(881, dtype=np.float32),
            })
            continue

        fp_maccs = np.asarray(
            list(AllChem.GetMACCSKeysFingerprint(mol_obj)),
            dtype=np.float32,
        )

        fp_erg = np.asarray(
            AllChem.GetErGFingerprint(
                mol_obj,
                fuzzIncrement=0.3,
                maxPath=21,
                minPath=1,
            ),
            dtype=np.float32,
        )

        fp_pubchem = np.asarray(
            GetPubChemFPs(mol_obj),
            dtype=np.float32,
        )

        fingerprints.append({
            "maccs": fp_maccs,
            "erg": fp_erg,
            "pubchem": fp_pubchem,
        })

    return fingerprints


def prepare_data(dataset, fingerprints, fp_molformer):
    if not (len(dataset.X) == len(fingerprints) == len(fp_molformer)):
        raise ValueError("Data length mismatch while preparing graph/molecule features.")

    data_list = []

    for i, (graph, y) in enumerate(zip(dataset.X, dataset.y)):
        x = torch.tensor(graph.node_features, dtype=torch.float32)
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long)

        fp_maccs = torch.tensor(
            fingerprints[i]["maccs"],
            dtype=torch.float32,
        )
        fp_erg = torch.tensor(
            fingerprints[i]["erg"],
            dtype=torch.float32,
        )
        fp_pubchem = torch.tensor(
            fingerprints[i]["pubchem"],
            dtype=torch.float32,
        )

        # Keep the original concatenated 1489-D fingerprint for prediction.
        fp = torch.cat(
            [fp_maccs, fp_erg, fp_pubchem],
            dim=0,
        )

        fp2 = torch.tensor(fp_molformer[i], dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            fp=fp.unsqueeze(0),
            fp_maccs=fp_maccs.unsqueeze(0),
            fp_erg=fp_erg.unsqueeze(0),
            fp_pubchem=fp_pubchem.unsqueeze(0),
            fp2=fp2.unsqueeze(0),
            y=y_tensor,
        )
        data_list.append(data)

    return data_list


def get_target_and_probs(outputs, batch_y, is_multilabel):
    if is_multilabel or outputs.shape[1] == 1:
        target = batch_y.view(-1, outputs.shape[1])
        probs = torch.sigmoid(outputs)
    else:
        target = batch_y.argmax(dim=1)
        probs = torch.softmax(outputs, dim=1)
    return target, probs


def split_model_output(model_output):
    if isinstance(model_output, (list, tuple)):
        aux_losses = model_output[1] if len(model_output) > 1 and isinstance(model_output[1], dict) else {}
        return model_output[-1], aux_losses
    return model_output, {}


def add_auxiliary_loss(
    task_loss,
    aux_losses,
    relation_loss_weight,
    contrastive_loss_weight,
):
    loss = task_loss

    if contrastive_loss_weight > 0 and "contrastive_loss" in aux_losses:
        loss = (
            loss
            + contrastive_loss_weight
            * aux_losses["contrastive_loss"]
        )

    if relation_loss_weight > 0 and "relation_loss" in aux_losses:
        loss = (
            loss
            + relation_loss_weight
            * aux_losses["relation_loss"]
        )

    return loss


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    is_multilabel,
    relation_loss_weight,
    contrastive_loss_weight,
):
    model.train()
    total_loss = 0.0
    y_true, y_scores = [], []

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        model_output = model(batch)
        outputs, aux_losses = split_model_output(model_output)
        target, probs = get_target_and_probs(outputs, batch.y, is_multilabel)

        task_loss = criterion(outputs, target)
        loss = add_auxiliary_loss(
            task_loss,
            aux_losses,
            relation_loss_weight,
            contrastive_loss_weight,
        )
        loss.backward()
        optimizer.step()

        total_loss += task_loss.item()
        y_scores.extend(probs.cpu().detach().numpy())
        y_true.extend(target.cpu().detach().numpy())

    prc, roc = compute_auc(y_true, y_scores)
    return total_loss / len(train_loader), prc, roc


def validate(model, loader, criterion, device, is_multilabel):
    model.eval()
    total_loss = 0.0
    y_true, y_scores = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs, _ = split_model_output(model(batch))
            target, probs = get_target_and_probs(outputs, batch.y, is_multilabel)
            loss = criterion(outputs, target)

            total_loss += loss.item()
            y_scores.extend(probs.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    prc, roc = compute_auc(y_true, y_scores)
    return total_loss / len(loader), prc, roc


def run_experiment(loader_func, dataset_name, model_save_path, cfg, device, log_fp):
    torch.cuda.empty_cache()
    _, datasets, _ = loader_func(
        featurizer=dc.feat.MolGraphConvFeaturizer(),
        splitter="random",
        splitter_seed=cfg.seed,
        reload=True,
        data_dir="./datasets/raw/",
        save_dir="./datasets/featurized/",
    )
    train_dataset, valid_dataset, test_dataset = datasets

    expected_sizes = (len(train_dataset.X), len(valid_dataset.X), len(test_dataset.X))
    output_dir_path = os.path.join("./datasets/Molformer_OUTPUT", dataset_name)
    train_fp_molformer, valid_fp_molformer, test_fp_molformer = load_molformer_embeddings(
        output_dir_path=output_dir_path,
        use_molformer=cfg.use_molformer,
        expected_sizes=expected_sizes,
    )

    train_fp = extract_fp_and_smiles(train_dataset)
    valid_fp = extract_fp_and_smiles(valid_dataset)
    test_fp = extract_fp_and_smiles(test_dataset)

    train_data = prepare_data(
        train_dataset, train_fp, train_fp_molformer
    )
    valid_data = prepare_data(
        valid_dataset, valid_fp, valid_fp_molformer
    )
    test_data = prepare_data(
        test_dataset, test_fp, test_fp_molformer
    )

    num_classes = train_dataset.y.shape[1]
    is_multilabel = num_classes > 1 and (train_dataset.y.sum(axis=1) > 1).any()
    out_size = num_classes if is_multilabel else (1 if num_classes == 1 else num_classes)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)

    molformer_dim = train_fp_molformer.shape[1]
    model = MoDAF(
        graph_dim=train_data[0].x.size(1),
        fp_dim=train_data[0].fp.size(1),
        molformer_dim=molformer_dim,
        hidden_dim=cfg.embed_dim,
        output_dim=out_size,
        num_layers=cfg.num_layers,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss() if is_multilabel or out_size == 1 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.9, patience=cfg.scheduler_patience, verbose=True)

    best_val_auc = 0.0
    num_not_improve = 0

    log_message(
        f"Dataset={dataset_name}, use_molformer={cfg.use_molformer}, molformer_dim={molformer_dim}, "
        f"output_dim={out_size}, is_multilabel={is_multilabel}",
        log_fp,
    )

    for epoch in range(cfg.epochs):
        time_start = time.time()
        train_loss, train_prc, train_roc = train(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            is_multilabel,
            cfg.relation_loss_weight,
            cfg.contrastive_loss_weight,
        )
        valid_loss, valid_prc, valid_roc = validate(model, valid_loader, criterion, device, is_multilabel)
        test_loss, test_prc, test_roc = validate(model, test_loader, criterion, device, is_multilabel)

        train_auc = train_prc if "muv" in dataset_name else train_roc
        valid_auc = valid_prc if "muv" in dataset_name else valid_roc
        test_auc = test_prc if "muv" in dataset_name else test_roc
        time_cost = time.time() - time_start

        scheduler.step(valid_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        if valid_auc > best_val_auc:
            best_val_auc = valid_auc
            num_not_improve = 0
            torch.save(model, model_save_path)
        else:
            num_not_improve += 1

        log_message(
            f"Epoch {epoch + 1}/{cfg.epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Auc: {train_auc:.4f}, "
            f"Valid Loss: {valid_loss:.4f}, Valid Auc: {valid_auc:.4f}, Best Valid Auc: {best_val_auc:.4f}, "
            f"Test Auc: {test_auc:.4f}, LR: {current_lr:.6f}, Cost Time: {time_cost:.2f}s",
            log_fp,
        )

        if num_not_improve >= cfg.early_stop_patience:
            log_message(f"Early stop at epoch {epoch + 1}.", log_fp)
            break

    model = torch.load(model_save_path, map_location=device)
    test_loss, test_prc, test_roc = validate(model, test_loader, criterion, device, is_multilabel)
    test_auc = test_prc if "muv" in dataset_name else test_roc
    log_message(f"Final Test Loss: {test_loss:.4f}, Final Test Auc: {test_auc:.4f}", log_fp)


def parse_args():
    parser = argparse.ArgumentParser(description="MolDAF training for molecular property prediction")
    parser.add_argument("--model-name", type=str, default="MolDAF_M20")
    parser.add_argument("--run-times", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)  
    parser.add_argument("--relation-loss-weight", type=float, default=0.01) 
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.01)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-molformer", dest="use_molformer", action="store_true")
    parser.add_argument("--no-molformer", dest="use_molformer", action="store_false")
    parser.set_defaults(use_molformer=True)
    return parser.parse_args()


def main():
    cfg = parse_args()
    device = torch.device(cfg.device)

    result_path = os.path.join("./result_Cla", cfg.model_name)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs("./weights_Cla", exist_ok=True)

    seeds = list(range(2025, 2035))  

    for run_idx, seed in enumerate(seeds):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        file_name = os.path.join(result_path, f"result_{cfg.model_name}_seed_{seed}.txt")
        print(file_name)
        if os.path.exists(file_name) and not cfg.debug:
            print("File already exists, skipping...")
            continue

        with open(file_name, "w", encoding="utf-8") as log_fp:
            log_message(f"Run Seed: {seed}", log_fp)
            for dataset_name, loader in classification_loaders.items():
                model_save_path = (f"./weights_Cla/{cfg.model_name}_{dataset_name}_seed_{seed}.pth")
                log_message(f"\nRunning {cfg.model_name} on {dataset_name}",log_fp)
                run_experiment(loader,dataset_name,model_save_path,cfg,device,log_fp)


if __name__ == "__main__":
    main()
