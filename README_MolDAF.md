# MolDAF: Multimodal Molecular Property Prediction

MolDAF integrates molecular graphs, MACCS/ErG/PubChem fingerprints, and optional MoLFormer embeddings using cross-modal attention, contrastive learning, and molecular-relation consistency learning.

## Project Structure

```text
MolDAF-main/
├── main_Cla.py                     # MoleculeNet classification
├── main_Reg.py                     # Regression (RMSE)
├── main_Reg_mae.py                 # QM regression (MAE)
├── model.py                        # MolDAF model
├── prepare_molformer_embeddings.py # Optional MoLFormer embeddings
├── new_pubchemfp.py                # PubChem fingerprints
├── utils.py                        # Evaluation metrics
├── environment.yml                 # Conda environment
├── pretrained/
│   └── MoLFormer-XL-both-10pct/    # Local pretrained MoLFormer model
│       ├── config.json
│       ├── configuration_molformer.py
│       ├── modeling_molformer.py
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── FS-mol/
    ├── main_Cla.py                 # CHEMBL classification
    ├── model.py                    # MolDAF model
    ├── new_pubchemfp.py            # PubChem fingerprints
    ├── utils.py                    # ROC-AUC / PR-AUC
    └── raw/                        # FS dataset files
```

## Setup

From the directory containing `environment.yml`:

```bash
conda env create -f environment.yml
conda activate mola_code
```

## Run

**Pretrained MoLFormer embedding**

```bash
python prepare_molformer_embeddings.py
```

**MoleculeNet experiments**:

```bash
python main_Cla.py
python main_Reg.py 
python main_Reg_mae.py 
```

Classification: BACE, BBBP, ClinTox, SIDER, Tox21, MUV. Regression: Delaney, FreeSolv, Lipo, QM7, QM8, QM9.

**FS-mol experiments**:

The FS dataset is stored in `MolDAF-main/FS-mol/raw/`. The FS dataset files include `CHEMBL829401.csv`, `CHEMBL1614450.csv`, `CHEMBL1738078.csv`, `CHEMBL3215081.csv`, and `CHEMBL3888461.csv`.

```bash
cd FS-mol
python main_Cla.py 
```
