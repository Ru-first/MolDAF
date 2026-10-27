import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool

# Graph Encoder
class GINEncoder(nn.Module):
    def __init__(self, graph_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(graph_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = F.relu(self.input_proj(data.x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, data.edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        mean_x = global_mean_pool(x, data.batch)

        return mean_x

# Fingerprint Encoder
class FingerprintEncoder(nn.Module):
    def __init__(self, fp_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim * 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, fp):
        return self.net(fp.float())

# SMILES Encoder (Molformer)
class MolFormerEmbeddingEncoder(nn.Module):
    def __init__(self, molformer_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.enabled = molformer_dim > 0
        if self.enabled:
            self.net = nn.Sequential(
                nn.Linear(molformer_dim, hidden_dim * 2),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        else:
            self.net = None

    def forward(self, fp2, batch_size, device):
        if not self.enabled:
            return torch.zeros(batch_size, 0, device=device)
        return self.net(fp2.float())


class CrossModalFusion(nn.Module):
    """Cross-attention + residual/LayerNorm + concat/MLP fusion."""

    def __init__(self, input_dims, fusion_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_modalities = len(input_dims)

        while num_heads > 1 and fusion_dim % num_heads != 0:
            num_heads -= 1

        self.input_proj = nn.ModuleList(
            [nn.Linear(dim, fusion_dim) for dim in input_dims]
        )
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=fusion_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in input_dims
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(fusion_dim) for _ in input_dims]
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * self.num_modalities, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, modal_features):
        projected = [
            proj(feature) for proj, feature in zip(self.input_proj, modal_features)
        ]

        enhanced_features = []
        for idx, query_feature in enumerate(projected):
            if self.num_modalities == 1:
                enhanced_features.append(self.norms[idx](query_feature))
                continue

            other_features = [
                feature for j, feature in enumerate(projected) if j != idx
            ]
            kv = torch.stack(other_features, dim=1)
            query = query_feature.unsqueeze(1)
            context, _ = self.cross_attn[idx](
                query=query,
                key=kv,
                value=kv,
                need_weights=False,
            )
            enhanced = self.norms[idx](query_feature + context.squeeze(1))
            enhanced_features.append(enhanced)

        fusion_input = torch.cat(enhanced_features, dim=-1)
        task_feature = self.fusion_mlp(fusion_input)
        return task_feature, enhanced_features


class FingerprintRelationConsistencyLoss(nn.Module):
    """
    Fingerprint-based relation consistency loss.

    MACCS and PubChem are binary fingerprints.
    Pairwise molecular similarities are independently computed
    using the Tanimoto coefficient and fused with equal weights by default.

    ErG is retained in the prediction fingerprint representation,
    but is not used to construct the relation teacher.
    """

    def __init__(
        self,
        gamma=1.0,
        alpha_maccs=0.5,
        alpha_pubchem=0.5,
        eps=1e-8,
    ):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

        alpha_sum = alpha_maccs + alpha_pubchem
        if alpha_sum <= 0:
            raise ValueError("The sum of fingerprint similarity weights must be positive.")

        self.alpha_maccs = alpha_maccs / alpha_sum
        self.alpha_pubchem = alpha_pubchem / alpha_sum

    def _tanimoto(self, fp):
        fp = fp.float()
        dot = torch.matmul(fp, fp.t())
        sq_norm = (fp * fp).sum(dim=1, keepdim=True)
        sim = dot / (sq_norm + sq_norm.t() - dot + self.eps)
        return sim.clamp(min=0.0, max=1.0)

    def build_teacher_similarity(self, fp_maccs, fp_pubchem):
        s_maccs = self._tanimoto(fp_maccs)
        s_pubchem = self._tanimoto(fp_pubchem)

        s_teacher = (
            self.alpha_maccs * s_maccs
            + self.alpha_pubchem * s_pubchem
        )

        component_sims = {
            "maccs_similarity": s_maccs.detach(),
            "pubchem_similarity": s_pubchem.detach(),
        }
        return s_teacher.detach(), component_sims

    def forward(self, relation_feature, fp_maccs, fp_pubchem):
        if relation_feature.size(0) <= 1:
            return relation_feature.new_tensor(0.0)

        s_teacher, _ = self.build_teacher_similarity(
            fp_maccs,
            fp_pubchem,
        )

        z = F.normalize(relation_feature, p=2, dim=1)
        s_repr = (torch.matmul(z, z.t()) + 1.0) / 2.0

        batch_size = relation_feature.size(0)
        mask = ~torch.eye(
            batch_size,
            dtype=torch.bool,
            device=relation_feature.device,
        )

        weight = torch.pow(s_teacher.clamp(min=0.0), self.gamma)

        return (
            weight[mask]
            * (s_repr[mask] - s_teacher[mask]).pow(2)
        ).mean()


class MultiModalContrastiveLoss(nn.Module):
    """Pairwise in-batch contrastive alignment in projection space."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def _pair_loss(self, z_a, z_b):
        if z_a.size(0) <= 1:
            return z_a.new_tensor(0.0)

        z_a = F.normalize(z_a, p=2, dim=1)
        z_b = F.normalize(z_b, p=2, dim=1)
        logits = torch.matmul(z_a, z_b.t()) / self.temperature
        labels = torch.arange(z_a.size(0), device=z_a.device)

        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_ab + loss_ba)

    def forward(self, z_list):
        if len(z_list) < 2:
            return z_list[0].new_tensor(0.0) if z_list else torch.tensor(0.0)

        pair_losses = []
        for i in range(len(z_list)):
            for j in range(i + 1, len(z_list)):
                pair_losses.append(self._pair_loss(z_list[i], z_list[j]))

        return torch.stack(pair_losses).mean()


class MolDAF(nn.Module):
    def __init__(self, graph_dim, fp_dim, molformer_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.active_modalities = ["fp", "graph"]
        self.fp_encoder = FingerprintEncoder(fp_dim, hidden_dim)
        self.graph_encoder = GINEncoder(graph_dim, hidden_dim, num_layers)
        self.molformer_encoder = MolFormerEmbeddingEncoder(molformer_dim, hidden_dim)
        if molformer_dim > 0:
            self.active_modalities.insert(1, "molformer")

        input_dims = [hidden_dim for _ in self.active_modalities]

        # Projection heads are used only by the contrastive objective.
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fp_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.smiles_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.contrastive_loss = MultiModalContrastiveLoss(temperature=0.1)

        # Prediction path: Cross-Attention -> Residual + LayerNorm -> Concat -> MLP.
        self.cross_fusion = CrossModalFusion(
            input_dims,
            hidden_dim,
            num_heads=4,
            dropout=0.1,
        )

        # Fingerprint remains only the Tanimoto relation teacher.
        self.relation_modalities = ["graph"]
        if "molformer" in self.active_modalities:
            self.relation_modalities.insert(0, "molformer")
        relation_input_dim = hidden_dim * len(self.relation_modalities)

        self.relation_proj = nn.Sequential(
            nn.Linear(relation_input_dim, max(hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.LayerNorm(max(hidden_dim // 2, 1)),
            nn.Linear(max(hidden_dim // 2, 1), hidden_dim),
        )
        self.relation_loss = FingerprintRelationConsistencyLoss(gamma=1.0)

        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_modalities(self, data):
        # Prediction branch still uses the original concatenated fingerprint.
        fp = data.fp.view(data.fp.size(0), -1)

        # Relation branch keeps three fingerprint views separately.
        fp_maccs = data.fp_maccs.view(data.fp_maccs.size(0), -1)
        fp_erg = data.fp_erg.view(data.fp_erg.size(0), -1)
        fp_pubchem = data.fp_pubchem.view(data.fp_pubchem.size(0), -1)

        fp2 = data.fp2.view(data.fp2.size(0), -1)
        batch_size = fp.size(0)
        device = fp.device

        feature_dict = {
            "fp": self.fp_encoder(fp),
            "graph": self.graph_encoder(data),
        }
        if "molformer" in self.active_modalities:
            feature_dict["molformer"] = self.molformer_encoder(
                fp2, batch_size, device
            )

        modal_features = [
            feature_dict[name] for name in self.active_modalities
        ]

        fingerprint_views = {
            "maccs": fp_maccs,
            "erg": fp_erg,
            "pubchem": fp_pubchem,
        }

        return feature_dict, modal_features, fp, fingerprint_views

    def forward(self, data):
        feature_dict, modal_features, fp, fingerprint_views = self.encode_modalities(data)

        # Contrastive objective operates only in projection space.
        z_list = [
            self.graph_proj(feature_dict["graph"]),
            self.fp_proj(feature_dict["fp"]),
        ]
        if "molformer" in feature_dict:
            z_list.append(self.smiles_proj(feature_dict["molformer"]))
        contrastive_loss = self.contrastive_loss(z_list)

        # Main prediction path has no gate or adaptive modality reweighting.
        task_feature, enhanced_features = self.cross_fusion(modal_features)
        task_output = self.task_head(task_feature)

        enhanced_dict = {
            name: feature
            for name, feature in zip(self.active_modalities, enhanced_features)
        }

        # Relation consistency is computed from pre-fusion encoder outputs.
        # Graph/MolFormer provide the student relation representation, while
        # raw MACCS/PubChem fingerprints are used only to construct Tanimoto targets.
        relation_sources = [
            feature_dict[name] for name in self.relation_modalities
        ]
        relation_feature = self.relation_proj(
            torch.cat(relation_sources, dim=1)
        )
        relation_loss = self.relation_loss(
            relation_feature,
            fingerprint_views["maccs"],
            fingerprint_views["pubchem"],
        )

        teacher_similarity, component_sims = (
            self.relation_loss.build_teacher_similarity(
                fingerprint_views["maccs"],
                fingerprint_views["pubchem"],
            )
        )

        aux_losses = {
            "contrastive_loss": contrastive_loss,
            "relation_loss": relation_loss,
        }
        relation_inputs = {
            "relation_feature": relation_feature,
            "fp": fp,
            "fp_maccs": fingerprint_views["maccs"],
            "fp_erg": fingerprint_views["erg"],
            "fp_pubchem": fingerprint_views["pubchem"],
            "teacher_similarity": teacher_similarity,
            **component_sims,
        }

        return [
            task_feature,
            aux_losses,
            relation_inputs,
            task_output,
        ]
