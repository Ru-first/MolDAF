"""
Prepare MoLFormer embeddings for MoleculeNet classification and regression datasets.

This script computes train/valid/test MoLFormer embeddings and stores them at:

    datasets/Molformer_OUTPUT/<dataset_name>/Molformer_Emb_2025.h5

This version uses HuggingFace local MoLFormer model:

    ./pretrained/MoLFormer-XL-both-10pct

Expected model download command:

    hf download ibm/MoLFormer-XL-both-10pct \
      --local-dir ./pretrained/MoLFormer-XL-both-10pct
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Callable, Dict, Iterable, Tuple

import deepchem as dc
import h5py
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")


DEFAULT_MODEL_PATH = "./pretrained/MoLFormer-XL-both-10pct"
DEFAULT_OUTPUT_FILE = "Molformer_Emb_2025.h5"


DATASET_LOADERS: Dict[str, Callable] = {
    #"bace_classification": dc.molnet.load_bace_classification,
    #"bbbp": dc.molnet.load_bbbp,
    #"clintox": dc.molnet.load_clintox,
    #"hiv": dc.molnet.load_hiv,
    #"muv": dc.molnet.load_muv,
    #"sider": dc.molnet.load_sider,
    #"tox21": dc.molnet.load_tox21,
    #"toxcast": dc.molnet.load_toxcast,
    #"delaney": dc.molnet.load_delaney,
    #"freesolv": dc.molnet.load_freesolv,
    #"lipo": dc.molnet.load_lipo,
    #"qm7": dc.molnet.load_qm7,
    #"qm8": dc.molnet.load_qm8,
    "qm9": dc.molnet.load_qm9,
}


class HuggingFaceMolFormerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_PATH,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 278,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.max_length = max_length

        if not os.path.exists(model_name):
            raise FileNotFoundError(
                f"MoLFormer model path not found: {model_name}\n"
                "Download it first:\n"
                "  hf download ibm/MoLFormer-XL-both-10pct "
                "--local-dir ./pretrained/MoLFormer-XL-both-10pct"
            )

        print(f"[MoLFormer] Loading model from: {model_name}")
        print(f"[MoLFormer] Device: {self.device}")
        print(f"[MoLFormer] Batch size: {self.batch_size}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            deterministic_eval=True,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def encode_batch(self, smiles_batch: list[str]) -> np.ndarray:
        tokens = self.tokenizer(
            smiles_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            embedding = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return embedding.cpu().numpy().astype(np.float32)

    def encode(self, smiles_list: Iterable[str]) -> np.ndarray:
        all_embeddings = []
        smiles_list = [smi if isinstance(smi, str) else "" for smi in smiles_list]

        for start in range(0, len(smiles_list), self.batch_size):
            end = min(start + self.batch_size, len(smiles_list))
            batch = smiles_list[start:end]
            print(f"[MoLFormer] Encoding batch {start}:{end} / {len(smiles_list)}")
            batch_embeddings = self.encode_batch(batch)
            all_embeddings.append(batch_embeddings)

        if not all_embeddings:
            return np.zeros((0, 0), dtype=np.float32)

        return np.concatenate(all_embeddings, axis=0)


_embedder: HuggingFaceMolFormerEmbedder | None = None


def build_molformer_embedder(
    model_path: str,
    device: str | None,
    batch_size: int,
    max_length: int,
) -> HuggingFaceMolFormerEmbedder:
    return HuggingFaceMolFormerEmbedder(
        model_name=model_path,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )


def molformer_embed(smiles_list: Iterable[str]) -> np.ndarray:
    if _embedder is None:
        raise RuntimeError("MoLFormer embedder is not initialized.")
    return _embedder.encode(smiles_list)


def _extract_embeddings(smiles_list: Iterable[str], split_name: str) -> np.ndarray:
    smiles = list(smiles_list)
    start = time.time()

    print(f"[MoLFormer] Encoding {split_name:<5} split, size={len(smiles)} ...")
    embeddings = molformer_embed(smiles)

    elapsed = time.time() - start
    emb = np.asarray(embeddings, dtype=np.float32)

    print(f"[MoLFormer] {split_name:<5} done: shape={emb.shape}, time={elapsed:.2f}s")
    return emb


def _save_h5(
    output_path: str,
    train_fp: np.ndarray,
    valid_fp: np.ndarray,
    test_fp: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("train_fp", data=train_fp, compression="gzip")
        f.create_dataset("valid_fp", data=valid_fp, compression="gzip")
        f.create_dataset("test_fp", data=test_fp, compression="gzip")

    print(f"[Save] {output_path}")


def _load_split_datasets(loader_func: Callable, seed: int):
    _, datasets, _ = loader_func(
        featurizer=dc.feat.MolGraphConvFeaturizer(),
        splitter="random",
        splitter_seed=seed,
        reload=True,
        data_dir="./datasets/raw/",
        save_dir="./datasets/featurized/",
    )
    return datasets


def prepare_dataset_embeddings(dataset_name: str, loader_func: Callable, seed: int = 2025) -> str:
    print(f"\n========== Preparing: {dataset_name} ==========")

    train_ds, valid_ds, test_ds = _load_split_datasets(loader_func, seed=seed)

    train_fp = _extract_embeddings(train_ds.ids, "train")
    valid_fp = _extract_embeddings(valid_ds.ids, "valid")
    test_fp = _extract_embeddings(test_ds.ids, "test")

    output_dir = os.path.join("./datasets/Molformer_OUTPUT", dataset_name)
    output_path = os.path.join(output_dir, DEFAULT_OUTPUT_FILE)

    _save_h5(output_path, train_fp, valid_fp, test_fp)
    return output_path

def run_selected_datasets(dataset_names: list[str], seed: int = 2025) -> Tuple[int, int]:
    success = 0
    failed = 0

    for dataset_name in dataset_names:
        loader = DATASET_LOADERS.get(dataset_name)
        if loader is None:
            failed += 1
            print(f"[Error] Unknown dataset: {dataset_name}")
            continue

        try:
            prepare_dataset_embeddings(dataset_name, loader, seed=seed)
            success += 1
        except Exception as exc:
            failed += 1
            print(f"[Error] {dataset_name}: {exc}")

    print(f"\nCompleted. success={success}, failed={failed}")
    return success, failed


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MoLFormer embeddings for MoLA.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Local HuggingFace MoLFormer model directory.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=list(DATASET_LOADERS.keys()),
        help="Dataset names to prepare. Default: all configured datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="DeepChem splitter seed. Must match training scripts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="MoLFormer inference batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=278,
        help="MoLFormer tokenizer max length.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for MoLFormer inference, e.g. cuda, cuda:0, cpu. Default: auto.",
    )
    return parser.parse_args()


def main():
    global _embedder

    args = parse_args()

    _embedder = build_molformer_embedder(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    run_selected_datasets(
        dataset_names=args.datasets,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()