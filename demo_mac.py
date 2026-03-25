"""
LeWorldModel Demo on Mac (MPS/CPU)
Loads trained checkpoint, runs forward pass, and visualizes latent space + predictions.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Use MPS if available, else CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================
# 1. Load the trained model
# ============================================================
ckpt_path = Path.home() / ".stable_worldmodel" / "lewm_epoch_1_object.ckpt"
if not ckpt_path.exists():
    print(f"Checkpoint not found at {ckpt_path}")
    print("Available checkpoints:")
    for p in ckpt_path.parent.glob("lewm_*_object.ckpt"):
        print(f"  {p.name}")
    exit(1)

print(f"Loading checkpoint: {ckpt_path.name}")
model = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model = model.to(DEVICE)
model.eval()
model.requires_grad_(False)
print("Model loaded successfully!")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

# ============================================================
# 2. Load test data from the HDF5 dataset
# ============================================================
import stable_worldmodel as swm
import stable_pretraining as spt
from torchvision.transforms import v2 as transforms

dataset_path = Path.home() / ".stable_worldmodel" / "pusht_expert_train.h5"
dataset = swm.data.HDF5Dataset(
    "pusht_expert_train",
    keys_to_load=["pixels", "action", "proprio", "state"],
    keys_to_cache=["action", "proprio", "state"],
)

# Image preprocessing (same as training)
img_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    transforms.Resize(size=224),
])

print(f"Dataset: {len(dataset)} frames")

# ============================================================
# 3. Run encoding demo - encode a batch of observations
# ============================================================
print("\n--- Encoding Demo ---")

# Get a sequence of 4 frames (history_size=3 + 1 prediction target)
# The dataset uses frameskip=5, so the action encoder expects action_dim * frameskip = 2*5 = 10 dims
# We need to use the dataset's sequencing to get proper frame groups
num_frames = 4
frameskip = 5  # from config

# Get frames spaced by frameskip
base_idx = 100
indices = [base_idx + i * frameskip for i in range(num_frames)]
frames = []
actions = []
raw_frames = []

for i, idx in enumerate(indices):
    row = dataset.get_row_data(idx)
    pixels = row["pixels"]  # (H, W, 3) uint8
    raw_frames.append(pixels.copy())

    # Transform pixels
    pixels_t = img_transform(pixels)
    frames.append(pixels_t)

    # Collect frameskip actions into a single vector (like the training does)
    act_group = []
    for fs in range(frameskip):
        act_idx = base_idx + i * frameskip + fs
        if act_idx < len(dataset):
            r = dataset.get_row_data(act_idx)
            act_group.append(r["action"])
        else:
            act_group.append(np.zeros(2))
    # Concatenate frameskip actions: (frameskip * action_dim,) = (10,)
    actions.append(torch.tensor(np.concatenate(act_group), dtype=torch.float32))

# Stack into batch: (1, T, C, H, W)
pixels_batch = torch.stack(frames).unsqueeze(0).to(DEVICE)
actions_batch = torch.stack(actions).unsqueeze(0).to(DEVICE)

print(f"Input pixels shape: {pixels_batch.shape}")
print(f"Input actions shape: {actions_batch.shape}")

# Encode
info = {"pixels": pixels_batch, "action": actions_batch}
with torch.no_grad():
    start = time.time()
    output = model.encode(info)
    encode_time = time.time() - start

emb = output["emb"]
act_emb = output["act_emb"]
print(f"Embedding shape: {emb.shape} (dim={emb.shape[-1]})")
print(f"Action embedding shape: {act_emb.shape}")
print(f"Encoding time: {encode_time*1000:.1f}ms")

# ============================================================
# 4. Run prediction demo
# ============================================================
print("\n--- Prediction Demo ---")

ctx_emb = emb[:, :3]  # context: first 3 frames
ctx_act = act_emb[:, :3]

with torch.no_grad():
    start = time.time()
    pred_emb = model.predict(ctx_emb, ctx_act)
    pred_time = time.time() - start

target_emb = emb[:, 1:]  # ground truth shifted by 1
pred_loss = (pred_emb - target_emb).pow(2).mean().item()
print(f"Prediction loss (MSE): {pred_loss:.6f}")
print(f"Prediction time: {pred_time*1000:.1f}ms")

# ============================================================
# 5. Run planning/rollout demo
# ============================================================
print("\n--- Rollout Demo ---")

# Simulate a planning scenario
# Take first 3 frames as context, generate action candidates, rollout
num_samples = 64   # action plan candidates
horizon = 10       # planning steps
action_dim = 10    # PushT: 2 * frameskip=5 = 10

# Initial observation (3 frames)
init_pixels = pixels_batch[:, :3]  # (1, 3, C, H, W)
init_actions = actions_batch[:, :3]

# Random action candidates (1, S, T, action_dim)
action_candidates = torch.randn(1, num_samples, horizon, action_dim, device=DEVICE) * 0.3
# First 3 steps are the context actions
action_candidates[:, :, :3, :] = init_actions.unsqueeze(1).expand(-1, num_samples, -1, -1)

info_plan = {"pixels": init_pixels.unsqueeze(1).expand(-1, num_samples, -1, -1, -1, -1)}

with torch.no_grad():
    start = time.time()
    rollout_info = model.rollout(info_plan, action_candidates, history_size=3)
    rollout_time = time.time() - start

pred_rollout = rollout_info["predicted_emb"]
print(f"Rollout output shape: {pred_rollout.shape}")
print(f"Rollout time ({num_samples} candidates, {horizon} steps): {rollout_time*1000:.0f}ms")
print(f"Planning speed: {rollout_time:.2f}s (target: <1s)")

# ============================================================
# 6. Visualize
# ============================================================
print("\n--- Creating Visualizations ---")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Top row: Input frames
for i in range(min(4, len(raw_frames))):
    axes[0, i].imshow(raw_frames[i])
    axes[0, i].set_title(f"Frame {i}")
    axes[0, i].axis("off")

# Bottom left: Embedding similarity matrix
emb_np = emb[0].cpu().numpy()  # (T, D)
sim_matrix = np.corrcoef(emb_np)
im = axes[1, 0].imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
axes[1, 0].set_title("Embedding Similarity")
axes[1, 0].set_xlabel("Frame")
axes[1, 0].set_ylabel("Frame")
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# Bottom middle: Prediction error per timestep
pred_np = pred_emb[0].cpu().numpy()
target_np = target_emb[0].cpu().numpy()
errors = np.mean((pred_np - target_np) ** 2, axis=-1)
axes[1, 1].bar(range(len(errors)), errors, color='steelblue')
axes[1, 1].set_title("Prediction Error per Step")
axes[1, 1].set_xlabel("Step")
axes[1, 1].set_ylabel("MSE")

# Bottom right: Embedding PCA visualization
from sklearn.decomposition import PCA

# Collect embeddings from rollout
rollout_emb = pred_rollout[0].cpu().numpy()  # (S, T, D)
# Flatten for PCA
all_emb = rollout_emb.reshape(-1, rollout_emb.shape[-1])
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(all_emb)
emb_2d = emb_2d.reshape(num_samples, -1, 2)

for s in range(min(20, num_samples)):
    axes[1, 2].plot(emb_2d[s, :, 0], emb_2d[s, :, 1], alpha=0.3, linewidth=0.5)
axes[1, 2].set_title(f"Latent Rollout Trajectories\n(PCA, {min(20, num_samples)} samples)")
axes[1, 2].set_xlabel("PC1")
axes[1, 2].set_ylabel("PC2")

# Bottom far right: Embedding norms over rollout
mean_norms = np.linalg.norm(rollout_emb.mean(axis=0), axis=-1)
axes[1, 3].plot(mean_norms, 'o-', color='darkgreen')
axes[1, 3].set_title("Mean Embedding Norm\nover Rollout Steps")
axes[1, 3].set_xlabel("Step")
axes[1, 3].set_ylabel("L2 Norm")

plt.suptitle(f"LeWorldModel (LeWM) - PushT Demo on Mac\n"
             f"~{total_params/1e6:.1f}M params | Encode: {encode_time*1000:.0f}ms | "
             f"Predict: {pred_time*1000:.0f}ms | Rollout ({num_samples}×{horizon}): {rollout_time*1000:.0f}ms",
             fontsize=12, fontweight='bold')
plt.tight_layout()

output_path = Path(__file__).parent / "lewm_demo_results.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nResults saved to: {output_path}")

# ============================================================
# 7. Summary
# ============================================================
print("\n" + "="*60)
print("LeWorldModel (LeWM) - Demo Summary")
print("="*60)
print(f"  Model:          LeWM (JEPA, end-to-end)")
print(f"  Parameters:     {total_params:,} ({total_params/1e6:.1f}M)")
print(f"  Encoder:        ViT-Tiny (patch_size=14)")
print(f"  Embed dim:      {emb.shape[-1]}")
print(f"  Device:         {DEVICE}")
print(f"  Encoding time:  {encode_time*1000:.1f}ms (4 frames)")
print(f"  Prediction:     {pred_time*1000:.1f}ms (3-step)")
print(f"  Planning:       {rollout_time*1000:.0f}ms ({num_samples} candidates × {horizon} steps)")
print(f"  Pred MSE:       {pred_loss:.6f}")
print(f"  Training:       1 epoch on PushT (200 random episodes)")
print("="*60)
print("\nKey insight: LeWM uses just 2 losses:")
print("  1. MSE prediction loss (next-embedding prediction)")
print("  2. SIGReg (anti-collapse regularizer)")
print("No EMA, no stop-grad, no pretrained encoders needed!")
