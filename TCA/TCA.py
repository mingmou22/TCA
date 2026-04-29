import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import warnings
from typing import Optional, Dict, List, Tuple
from scipy import ndimage

warnings.filterwarnings('ignore')


def load_image(img_path, img_size=224, device='cpu'):
    img = Image.open(img_path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])
    tensor = tf(img).to(device)
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor


def compute_enhanced_features(img):
    
    img_np = img.permute(1, 2, 0).cpu().numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    edges_fine = cv2.Canny(gray, 30, 100)
    edges_coarse = cv2.Canny(gray, 100, 200)
    edges = np.maximum(edges_fine, edges_coarse).astype(np.float32) / 255.0

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    structure_score = np.abs(np.abs(sobelx) - np.abs(sobely)) / 255.0

    local_std = ndimage.generic_filter(gray.astype(np.float32), np.std, size=5)
    texture = local_std / (local_std.max() + 1e-8)

    corners = cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    corners = np.clip(corners / (corners.max() + 1e-8), 0, 1)

    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    large_scale = np.abs(gray.astype(np.float32) - blurred) / 255.0

    features = np.stack([edges, structure_score, texture, corners, large_scale], axis=0)
    return torch.from_numpy(features).float()



class MagnitudeEnhancedDecoder(nn.Module):
    
    def __init__(self, hidden_dim=128, content_dim=5):
        super().__init__()

        self.content_encoder = nn.Sequential(
            nn.Linear(content_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2)
        )

        combined_dim = hidden_dim + 32

        self.offset_net = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2)
        )

        self.scale_net = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2)
        )

        self.rotation_net = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )

        
        self.color_net = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 3)
        )

        
        self.color_direction = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  
        )

        self.alpha_net = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.alpha_net.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.uniform_(m.bias, -0.3, 0.8)

    def forward(self, node_features, content_features):
        content_encoded = self.content_encoder(content_features)
        combined = torch.cat([node_features, content_encoded], dim=-1)

        offset_raw = self.offset_net(combined)
        scale_raw = self.scale_net(combined)
        rotation_raw = self.rotation_net(combined)
        color_raw = self.color_net(combined)
        color_dir = self.color_direction(combined)  
        alpha_raw = self.alpha_net(combined).squeeze(-1)

        
        edges = content_features[:, 0]
        structure = content_features[:, 1]
        corners = content_features[:, 3]

        
        color_magnitude = torch.tanh(color_raw) * 0.8  
        
        color = color_magnitude * color_dir  

        
        content_importance = torch.clamp(edges + structure + corners, 0, 1)

        
        alpha_base = torch.sigmoid(alpha_raw)
        alpha = 0.15 + 0.8 * content_importance + 0.1 * alpha_base
        alpha = torch.clamp(alpha, 0.15, 0.95)

        
        scale_base = torch.clamp(F.softplus(scale_raw) * 3 + 3, min=2, max=20)
        scale_factor = 1.2 - 0.6 * content_importance.unsqueeze(-1)
        scale = scale_base * scale_factor
        scale = torch.clamp(scale, 2, 20)

        rotation = rotation_raw.squeeze(-1) * np.pi * 2

        return {
            'offset': torch.tanh(offset_raw) * 14,
            'scale': scale,
            'rotation': rotation,
            'color': color,  
            'alpha': alpha,
            'content_importance': content_importance,
            'color_direction': color_dir  
        }

class MagnitudeRasterizer2D(nn.Module):
    def __init__(self, image_size=(224, 224)):
        super().__init__()
        self.H, self.W = image_size

    def forward(self, gaussian_params, base_positions, importance_weights=None):
        device = base_positions.device
        N = base_positions.shape[0]

        alpha = gaussian_params['alpha']
        colors = gaussian_params['color']
        color_dir = gaussian_params.get('color_direction', torch.ones(N, 1, device=device))

        print(f"Rasterizer - Alpha: [{alpha.min():.3f}, {alpha.max():.3f}], σ={alpha.std():.3f}")
        print(f"  Color dir: [{color_dir.min():.3f}, {color_dir.max():.3f}] (should be -1 or 1)")
        print(f"  Color range: [{colors.min():.3f}, {colors.max():.3f}]")

        centers = base_positions + gaussian_params['offset']
        centers = torch.clamp(centers, 0, max(self.H, self.W))

        scales = gaussian_params['scale']
        rotations = gaussian_params['rotation']

        cos_r, sin_r = torch.cos(rotations), torch.sin(rotations)
        R = torch.stack([
            torch.stack([cos_r, -sin_r], dim=1),
            torch.stack([sin_r, cos_r], dim=1)
        ], dim=1)

        S = torch.diag_embed(scales)
        cov = R @ S @ S.transpose(-2, -1) @ R.transpose(-2, -1)
        cov_inv = torch.inverse(cov + torch.eye(2, device=device) * 1e-4)

        y_coords = torch.arange(self.H, device=device).float()
        x_coords = torch.arange(self.W, device=device).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixels = torch.stack([xx, yy], dim=-1)

        diff = pixels.unsqueeze(0) - centers.view(N, 1, 1, 2)
        dist = torch.einsum('nhwi,nij,nhwj->nhw', diff, cov_inv, diff)
        weights = torch.exp(-0.5 * dist.clamp(max=30))

        if importance_weights is not None:
            alpha = alpha * importance_weights.view(-1)

        
        weighted_color = torch.einsum('nhw,n,nk->hwk', weights, alpha, colors)
        perturbation = weighted_color

        
        pert_mean_per_channel = perturbation.mean(dim=(0,1))
        print(f"  Perturbation per channel mean: R={pert_mean_per_channel[0]:.4f}, "
              f"G={pert_mean_per_channel[1]:.4f}, B={pert_mean_per_channel[2]:.4f}")

        pert_std = perturbation.std(dim=(0,1)).mean()
        print(f"  Perturbation total range: [{perturbation.min():.4f}, {perturbation.max():.4f}]")

        return perturbation.permute(2, 0, 1).unsqueeze(0)



class MagnitudeEnhancedGNN(nn.Module):
    def __init__(self, in_dim, hidden=128, K=2, n_layers=3, content_dim=5):
        super().__init__()
        self.n_layers = n_layers

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2)
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(ChebConv(hidden, hidden, K))
            self.norms.append(nn.LayerNorm(hidden))

        self.gaussian_decoder = MagnitudeEnhancedDecoder(hidden, content_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, ChebConv):
                for lin in m.lins:
                    nn.init.kaiming_normal_(lin.weight, a=0.2)
                    if lin.bias is not None:
                        nn.init.zeros_(lin.bias)

    def forward(self, data, content_features, debug=False):
        x, edge_index = data.x, data.edge_index
        h = self.input_proj(x)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.leaky_relu(h_new, 0.2)

            if i > 0 and i % 2 == 1:
                h = h + h_new
            else:
                h = h_new

        gaussian_params = self.gaussian_decoder(h, content_features)
        score = h.std(dim=1)
        return gaussian_params, score, h


def image_to_patches(img, patch_size=16, stride=8, mask=None):
    
    C, H, W = img.shape
    device = img.device

    content_global = compute_enhanced_features(img).to(device)

    node_feats, idx_map, content_feats = [], [], []

    gid = 0
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            if mask is not None:
                m_patch = mask[i:i + patch_size, j:j + patch_size]
                if m_patch.sum() < patch_size * patch_size * 0.3:
                    continue

            patch = img[:, i:i + patch_size, j:j + patch_size]
            node_feats.append(patch.reshape(-1))

            center_i = i + patch_size // 2
            center_j = j + patch_size // 2
            content_patch = content_global[:, center_i, center_j]
            content_feats.append(content_patch)

            idx_map.append((i, j, gid, patch_size))
            gid += 1

    if len(node_feats) == 0:
        raise RuntimeError("No valid patches found")

    node_feats = torch.stack(node_feats)
    content_feats = torch.stack(content_feats)

    edge_list = []
    coord2idx = {(i, j): idx for idx, (i, j, _, _) in enumerate(idx_map)}

    for idx, (i, j, _, _) in enumerate(idx_map):
        for dx, dy in [(-stride, 0), (0, -stride), (stride, 0), (0, stride)]:
            ni, nj = i + dx, j + dy
            if (ni, nj) in coord2idx:
                edge_list.append((idx, coord2idx[(ni, nj)]))

    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).T
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    return node_feats, edge_index, idx_map, content_feats


def magnitude_enhanced_attack(
    img: torch.Tensor,
    structural_gnn: nn.Module,
    device: torch.device,
    patch_size: int = 16,
    stride: int = 8,
    n_steps: int = 400,
    adv_eps: float = 0.08,
    lr: float = 0.03,
    verbose: bool = True,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor, torch.Tensor, torch.Tensor]:

    img = img.to(device)
    C, H, W = img.shape

    node_feats, edge_index, idx_map, content_feats = image_to_patches(
        img, patch_size=patch_size, stride=stride, mask=mask
    )

    N = node_feats.shape[0]
    print(f"Magnitude enhanced attack: {N} nodes")

    base_positions = torch.tensor([
        [j + patch_size/2.0, i + patch_size/2.0]
        for i, j, _, _ in idx_map
    ], device=device).float()

    G = nx.Graph()
    G.add_nodes_from(range(N))
    if edge_index.shape[1] > 0:
        edges = edge_index.cpu().numpy().T
        G.add_edges_from([tuple(e) for e in edges])
        centrality = nx.degree_centrality(G)
    else:
        centrality = {i: 1.0 for i in range(N)}

    centrality_tensor = torch.tensor([centrality[i] for i in range(N)], device=device)
    imp_weight = F.softmax(centrality_tensor / 0.5, dim=0)

    node_feats = node_feats.to(device)
    edge_index = edge_index.to(device)
    content_feats = content_feats.to(device)
    data = Data(x=node_feats, edge_index=edge_index)

    optimizer = optim.Adam(structural_gnn.parameters(), lr=lr, betas=(0.9, 0.999))

    rasterizer = MagnitudeRasterizer2D(image_size=(H, W)).to(device)
    structural_gnn.train()

    best_perturbation = None
    best_params = None
    best_magnitude = 0

    for step in range(n_steps):
        optimizer.zero_grad()

        gaussian_params, score, node_h = structural_gnn(data, content_feats)
        perturbation = rasterizer(gaussian_params, base_positions, imp_weight)

        alpha = gaussian_params['alpha']
        colors = gaussian_params['color']

        if torch.isnan(perturbation).any():
            continue

        
        pert_magnitude = perturbation.abs().mean()
        pert_max = perturbation.abs().max()
        pert_std = perturbation.std()

       
        perturbation_clamped = torch.clamp(perturbation, -adv_eps, adv_eps)
        budget_penalty = (perturbation.abs() - adv_eps).clamp(min=0).mean() * 10.0

        
        magnitude_reward = pert_magnitude * 100.0  
        max_reward = pert_max * 50.0  

        perturbation = torch.clamp(perturbation, -adv_eps, adv_eps)
        adv_img_batch = torch.clamp(img.unsqueeze(0) + perturbation, 0, 1)

        
        edge_map = content_feats[:, 0]
        structure_map = content_feats[:, 1]
        corner_map = content_feats[:, 3]
        structure_weight = (edge_map + structure_map + corner_map * 2).clamp(0, 1)
        structure_alignment = (perturbation.abs().mean(dim=(1,2,3)) * structure_weight).mean()
        structure_loss = -structure_alignment * 40.0

        
        spatial_loss = -pert_std * 60.0
        alpha_std_loss = -alpha.std() * 40.0
        color_std_loss = -colors.std(dim=0).mean() * 20.0

       
        content_align_loss = -(alpha * structure_weight).mean() * 25.0

        loss = (structure_loss + spatial_loss + alpha_std_loss + color_std_loss +
                content_align_loss + magnitude_reward + max_reward + budget_penalty)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(structural_gnn.parameters(), max_norm=15.0)
        optimizer.step()

        with torch.no_grad():
            if pert_magnitude > best_magnitude:
                best_magnitude = pert_magnitude
                best_perturbation = perturbation.squeeze(0).clone()
                best_params = {k: v.clone() for k, v in gaussian_params.items()}

        if verbose and step % 40 == 0:
            print(f"[{step:03d}/{n_steps}] Loss: {loss.item():.4f} | "
                  f"Pert magnitude: {pert_magnitude:.4f} | Max: {pert_max:.4f} | "
                  f"Alpha σ: {alpha.std():.3f}")

    structural_gnn.eval()
    with torch.no_grad():
        if best_perturbation is not None:
            final_perturbation = best_perturbation
            final_params = best_params
            print(f"\nUsing best (magnitude: {best_magnitude:.4f})")
        else:
            gaussian_params, _, _ = structural_gnn(data, content_feats)
            final_perturbation = rasterizer(gaussian_params, base_positions, imp_weight)
            final_perturbation = torch.clamp(final_perturbation.squeeze(0), -adv_eps, adv_eps)
            final_params = gaussian_params
            print("\nUsing final")

        final_adv_img = torch.clamp(img + final_perturbation, 0, 1)

        actual_diff = (final_adv_img - img).abs()
        alphas = final_params['alpha']

        print(f"\n{'='*60}")
        print("MAGNITUDE ENHANCED RESULTS")
        print(f"{'='*60}")
        print(f"Perturbation L∞: {actual_diff.max():.6f} (target: {adv_eps})")
        print(f"Perturbation mean magnitude: {final_perturbation.abs().mean():.6f}")
        print(f"Alpha: [{alphas.min():.3f}, {alphas.max():.3f}], σ={alphas.std():.3f}")

    return final_adv_img, final_perturbation, final_params, base_positions, imp_weight, content_feats



def visualize_fixed_results(image, adv_img, perturbation, gaussian_params, base_positions, content_feats,
                             adv_eps=0.08, save_path='fixed_visualization.png'):

    if image.dim() == 4:
        image = image.squeeze(0)
    if adv_img.dim() == 4:
        adv_img = adv_img.squeeze(0)
    if perturbation.dim() == 4:
        perturbation = perturbation.squeeze(0)

    img_np = image.permute(1, 2, 0).cpu().numpy()
    adv_np = adv_img.permute(1, 2, 0).cpu().numpy()

    pert_np = perturbation.detach().cpu().numpy()
    if pert_np.shape[0] == 3:
        pert_np = pert_np.transpose(1, 2, 0)

   
    H, W = img_np.shape[:2]
    patch_size = 16
    stride = 8

    importance_img = np.zeros((H, W))
    count = 0
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            if count < len(content_feats):
                imp = content_feats[count, 0] + content_feats[count, 1] + content_feats[count, 3] * 2
                importance_img[i:i+patch_size, j:j+patch_size] = imp.cpu().clamp(0, 1)
                count += 1

    diff = np.abs(adv_np - img_np)
    diff_max = diff.max()

    
    diff_r = np.abs(adv_np[:,:,0] - img_np[:,:,0])
    diff_g = np.abs(adv_np[:,:,1] - img_np[:,:,1])
    diff_b = np.abs(adv_np[:,:,2] - img_np[:,:,2])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    
    axes[0, 0].imshow(np.clip(img_np, 0, 1))
    axes[0, 0].set_title('Original', fontweight='bold')
    axes[0, 0].axis('off')

   
    axes[0, 1].imshow(np.clip(adv_np, 0, 1))
    axes[0, 1].set_title('Adversarial', fontweight='bold')
    axes[0, 1].axis('off')

    
    im_imp = axes[0, 2].imshow(importance_img, cmap='hot')
    axes[0, 2].set_title('Content Importance', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im_imp, ax=axes[0, 2], fraction=0.046)

    
    pert_mean = pert_np.mean(axis=2) if pert_np.shape[2] == 3 else pert_np
    pert_min, pert_max = pert_mean.min(), pert_mean.max()

   
    display_range = max(abs(pert_min), abs(pert_max), 0.001)
    im_pert = axes[0, 3].imshow(pert_mean, cmap='RdBu_r',
                                 vmin=-display_range, vmax=display_range)
    title_color = 'green' if (pert_max - pert_min) > 0.001 else 'red'
    axes[0, 3].set_title(f'Perturbation Mean\n[{pert_min:.4f}, {pert_max:.4f}]',
                        fontweight='bold', color=title_color)
    axes[0, 3].axis('off')
    plt.colorbar(im_pert, ax=axes[0, 3], fraction=0.046)

    
    if pert_np.shape[2] == 3:
        pert_l2 = np.sqrt(np.sum(pert_np ** 2, axis=2))
    else:
        pert_l2 = np.abs(pert_np)

    im_l2 = axes[1, 0].imshow(pert_l2, cmap='viridis')
    axes[1, 0].set_title(f'Perturbation L2\n[{pert_l2.min():.4f}, {pert_l2.max():.4f}]', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im_l2, ax=axes[1, 0], fraction=0.046)


    diff_color = np.stack([diff_r, diff_g, diff_b], axis=2)
    diff_color = diff_color / (diff_color.max() + 1e-8)  

    axes[1, 1].imshow(diff_color)
    axes[1, 1].set_title(f'Difference (RGB)\nMax: {diff_max:.4f}', fontweight='bold')
    axes[1, 1].axis('off')

    
    axes[1, 2].imshow(importance_img, cmap='gray', alpha=0.3)
    axes[1, 2].imshow(np.clip(img_np, 0, 1), alpha=0.2)

    centers = (base_positions + gaussian_params['offset']).detach().cpu().numpy()
    scales = gaussian_params['scale'].detach().cpu().numpy()
    rotations = gaussian_params['rotation'].detach().cpu().numpy()
    alphas = gaussian_params['alpha'].detach().cpu().numpy()

    edge_vals = content_feats[:, 0].cpu().numpy()

    active = high = normal = 0
    for i in range(len(centers)):
        if alphas[i] > 0.15:
            active += 1
            alpha_val = alphas[i]
            is_edge = edge_vals[i] > 0.1 if i < len(edge_vals) else False

            if is_edge and alpha_val > 0.7:
                color = 'red'
                lw = 3.0
                high += 1
            else:
                color = 'lime'
                lw = 1.0
                normal += 1

            ellipse = Ellipse(
                xy=centers[i],
                width=float(scales[i, 0] * 2),
                height=float(scales[i, 1] * 2),
                angle=float(np.degrees(rotations[i])),
                facecolor='none',
                edgecolor=color,
                alpha=min(float(alpha_val), 1.0),
                linewidth=lw
            )
            axes[1, 2].add_patch(ellipse)

    axes[1, 2].set_title(f'Gaussians: {active} active\n{high} high, {normal} normal', fontweight='bold')
    axes[1, 2].set_xlim(0, image.shape[2])
    axes[1, 2].set_ylim(image.shape[1], 0)
    axes[1, 2].axis('off')

    # 8. Alpha distribution
    edge_alphas = [alphas[i] for i in range(len(alphas)) if i < len(edge_vals) and edge_vals[i] > 0.1]
    normal_alphas = [alphas[i] for i in range(len(alphas)) if i >= len(edge_vals) or edge_vals[i] <= 0.1]

    axes[1, 3].hist(edge_alphas, bins=20, range=(0, 1), color='red', alpha=0.7, label=f'Edge ({len(edge_alphas)})')
    axes[1, 3].hist(normal_alphas, bins=20, range=(0, 1), color='green', alpha=0.5, label=f'Normal ({len(normal_alphas)})')
    axes[1, 3].axvline(x=alphas.mean(), color='blue', linestyle='-', linewidth=2, label=f'μ={alphas.mean():.2f}')
    axes[1, 3].set_title(f'Alpha Distribution\nσ={alphas.std():.3f}', fontweight='bold')
    axes[1, 3].set_xlabel('Alpha Value')
    axes[1, 3].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fixed Visualization Attack on: {device}")

    img_path = '200.jpg'
    img_size = 224
    adv_eps = 0.08

    try:
        try:
            image = load_image(img_path, img_size=img_size, device=device)
        except:
            print("Using random image")
            torch.manual_seed(42)
            image = torch.rand(3, img_size, img_size).to(device) * 0.5 + 0.25

        print(f"Image: {image.shape}")

        in_dim = 3 * 16 * 16
        gnn = MagnitudeEnhancedGNN(in_dim=in_dim, hidden=128, K=2, n_layers=3, content_dim=5).to(device)

        print(f"\n{'='*60}")
        print("FIXED: Color direction alignment + Enhanced magnitude")
        print("Goal: Non-uniform Perturbation Mean & Difference")
        print(f"{'='*60}")

        adv_img, pert, params, pos, imp, content = magnitude_enhanced_attack(
            image,
            gnn,
            device,
            patch_size=16,
            stride=4,
            n_steps=200,
            adv_eps=adv_eps,
            lr=0.03,
            verbose=True
        )

        visualize_fixed_results(image, adv_img, pert, params, pos, content, adv_eps=adv_eps)

        adv_uint8 = (adv_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite("fixed_visualization_adversarial.png", cv2.cvtColor(adv_uint8, cv2.COLOR_RGB2BGR))

        print(f"\nSaved: fixed_visualization_adversarial.png")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()