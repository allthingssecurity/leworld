"""
LeWorldModel - Animated Inference Demo
Runs the trained JEPA world model and creates animations showing:
1. Real vs predicted latent rollouts
2. Latent space trajectories evolving over time
3. Planning with action candidates
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pathlib import Path
import time

import stable_worldmodel as swm
import stable_pretraining as spt
from torchvision.transforms import v2 as transforms
from sklearn.decomposition import PCA

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ============================================================
# Load best checkpoint
# ============================================================
ckpt_dir = Path.home() / ".stable_worldmodel"
ckpts = sorted(ckpt_dir.glob("lewm_epoch_*_object.ckpt"))
ckpt_path = ckpts[-1]  # best = latest epoch
print(f"Loading: {ckpt_path.name}")

model = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model = model.to(DEVICE).eval()
model.requires_grad_(False)
params = sum(p.numel() for p in model.parameters())
print(f"Model: {params/1e6:.1f}M params")

# ============================================================
# Load dataset
# ============================================================
dataset = swm.data.HDF5Dataset(
    "pusht_expert_train",
    keys_to_load=["pixels", "action", "proprio", "state"],
    keys_to_cache=["action", "proprio", "state"],
)

img_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    transforms.Resize(size=224),
])

FRAMESKIP = 5
ACTION_DIM = 2

def get_sequence(start_idx, num_frames):
    """Get a sequence of frames and actions from the dataset."""
    frames_raw = []
    frames_t = []
    actions = []

    for i in range(num_frames):
        idx = start_idx + i * FRAMESKIP
        if idx >= len(dataset):
            break
        row = dataset.get_row_data(idx)
        frames_raw.append(row["pixels"].copy())
        frames_t.append(img_transform(row["pixels"]))

        # Collect frameskip actions
        act_group = []
        for fs in range(FRAMESKIP):
            aidx = start_idx + i * FRAMESKIP + fs
            if aidx < len(dataset):
                r = dataset.get_row_data(aidx)
                act_group.append(r["action"])
            else:
                act_group.append(np.zeros(ACTION_DIM))
        actions.append(np.concatenate(act_group))

    pixels = torch.stack(frames_t).unsqueeze(0).to(DEVICE)
    acts = torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return frames_raw, pixels, acts


# ============================================================
# Animation 1: Latent space encoding over a trajectory
# ============================================================
print("\n--- Animation 1: Latent Trajectory ---")

num_frames = 30
start = 500  # pick a trajectory segment
raw_frames, pixels, actions = get_sequence(start, num_frames)

# Encode all frames
with torch.no_grad():
    info = {"pixels": pixels, "action": actions}
    output = model.encode(info)
    all_emb = output["emb"][0].cpu().numpy()  # (T, 192)

# PCA on embeddings
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(all_emb)

fig = plt.figure(figsize=(14, 5))
gs = GridSpec(1, 3, width_ratios=[1, 1.2, 1.2], figure=fig)
ax_img = fig.add_subplot(gs[0])
ax_latent = fig.add_subplot(gs[1])
ax_pred = fig.add_subplot(gs[2])

# Static setup
ax_latent.set_xlim(emb_2d[:, 0].min() - 0.5, emb_2d[:, 0].max() + 0.5)
ax_latent.set_ylim(emb_2d[:, 1].min() - 0.5, emb_2d[:, 1].max() + 0.5)
ax_latent.set_title("Latent Space (PCA)", fontsize=11)
ax_latent.set_xlabel("PC1")
ax_latent.set_ylabel("PC2")

# Plot full trajectory as faint line
ax_latent.plot(emb_2d[:, 0], emb_2d[:, 1], 'k-', alpha=0.1, linewidth=1)

colors = plt.cm.viridis(np.linspace(0, 1, num_frames))

def animate_latent(frame_idx):
    ax_img.clear()
    ax_img.imshow(raw_frames[frame_idx])
    ax_img.set_title(f"PushT Frame {frame_idx}/{num_frames-1}", fontsize=11)
    ax_img.axis("off")

    # Draw trajectory up to current frame
    if frame_idx > 0:
        ax_latent.plot(
            emb_2d[frame_idx-1:frame_idx+1, 0],
            emb_2d[frame_idx-1:frame_idx+1, 1],
            '-', color=colors[frame_idx], linewidth=2, alpha=0.8
        )
    ax_latent.plot(emb_2d[frame_idx, 0], emb_2d[frame_idx, 1],
                   'o', color=colors[frame_idx], markersize=8, zorder=5)

    # Prediction: encode context, predict next
    ax_pred.clear()
    if frame_idx >= 3:
        ctx_start = frame_idx - 3
        ctx_emb = output["emb"][:, ctx_start:frame_idx].to(DEVICE)
        ctx_act = output["act_emb"][:, ctx_start:frame_idx].to(DEVICE)
        with torch.no_grad():
            pred = model.predict(ctx_emb, ctx_act)

        pred_np = pred[0].cpu().numpy()
        target_np = all_emb[ctx_start+1:frame_idx+1]

        errors = np.sqrt(np.mean((pred_np - target_np) ** 2, axis=-1))
        bars = ax_pred.bar(range(len(errors)), errors, color=['#2196F3', '#4CAF50', '#FF9800'][:len(errors)])
        ax_pred.set_title(f"Prediction Error (RMSE)", fontsize=11)
        ax_pred.set_xlabel("Step in context")
        ax_pred.set_ylabel("RMSE")
        ax_pred.set_ylim(0, max(0.3, errors.max() * 1.2))
    else:
        ax_pred.text(0.5, 0.5, f"Waiting for context...\n({frame_idx+1}/3 frames)",
                     ha='center', va='center', fontsize=12, transform=ax_pred.transAxes)
        ax_pred.set_title("Prediction Error", fontsize=11)

    return []

print("  Rendering latent trajectory animation...")
anim1 = animation.FuncAnimation(fig, animate_latent, frames=num_frames, interval=300, blit=False)
anim1.save("lewm_latent_trajectory.gif", writer="pillow", fps=4, dpi=100)
plt.close()
print("  Saved: lewm_latent_trajectory.gif")


# ============================================================
# Animation 2: Planning rollout visualization
# ============================================================
print("\n--- Animation 2: Planning Rollout ---")

# Get a starting state (3 context frames)
ctx_frames_raw, ctx_pixels, ctx_actions = get_sequence(1000, 3)

# Also get ground truth future (20 frames ahead)
future_raw, future_pixels, future_actions = get_sequence(1000, 23)

# Encode context
with torch.no_grad():
    ctx_info = {"pixels": ctx_pixels, "action": ctx_actions}
    ctx_output = model.encode(ctx_info)
    ctx_emb = ctx_output["emb"]  # (1, 3, 192)
    ctx_act_emb = ctx_output["act_emb"]

    # Encode full future for ground truth
    fut_info = {"pixels": future_pixels, "action": future_actions}
    fut_output = model.encode(fut_info)
    gt_emb = fut_output["emb"][0].cpu().numpy()  # (23, 192)

# Generate multiple action candidates
num_samples = 128
horizon = 20
rng = np.random.default_rng(42)

# Mix of random and slightly-perturbed real actions
real_acts = future_actions[0, 3:23].cpu().numpy()  # ground truth future actions
action_candidates = []
for _ in range(num_samples):
    noise_scale = rng.uniform(0.0, 2.0)
    noisy = real_acts + rng.normal(0, noise_scale, real_acts.shape)
    noisy = np.clip(noisy, -1, 1)
    action_candidates.append(noisy)

action_candidates = torch.tensor(np.array(action_candidates), dtype=torch.float32).unsqueeze(0).to(DEVICE)
# Prepend context actions: (1, S, 3+20, act_dim)
ctx_acts_expanded = ctx_actions.unsqueeze(1).expand(-1, num_samples, -1, -1)
full_actions = torch.cat([ctx_acts_expanded, action_candidates], dim=2)

# Run rollout
init_pixels = ctx_pixels.unsqueeze(1).expand(-1, num_samples, -1, -1, -1, -1)
plan_info = {"pixels": init_pixels}

with torch.no_grad():
    t0 = time.time()
    rollout = model.rollout(plan_info, full_actions, history_size=3)
    rollout_time = time.time() - t0

pred_emb = rollout["predicted_emb"][0].cpu().numpy()  # (S, T, 192)
print(f"  Rollout: {num_samples} candidates × {horizon+3} steps in {rollout_time:.2f}s")

# PCA on all embeddings (ground truth + predicted)
all_for_pca = np.vstack([gt_emb, pred_emb.reshape(-1, pred_emb.shape[-1])])
pca2 = PCA(n_components=2)
all_2d = pca2.fit_transform(all_for_pca)
gt_2d = all_2d[:len(gt_emb)]
pred_2d = all_2d[len(gt_emb):].reshape(num_samples, -1, 2)

fig2 = plt.figure(figsize=(14, 5))
gs2 = GridSpec(1, 3, width_ratios=[1, 1.5, 1], figure=fig2)
ax_frame = fig2.add_subplot(gs2[0])
ax_plan = fig2.add_subplot(gs2[1])
ax_cost = fig2.add_subplot(gs2[2])

# Compute costs for each candidate
costs = np.mean((pred_emb[:, -1, :] - gt_emb[-1:, :]) ** 2, axis=-1)
best_idx = np.argmin(costs)
worst_idx = np.argmax(costs)

def animate_planning(step):
    ax_frame.clear()
    frame_idx = min(step + 3, len(future_raw) - 1)
    ax_frame.imshow(future_raw[frame_idx])
    ax_frame.set_title(f"Ground Truth (t={frame_idx})", fontsize=11)
    ax_frame.axis("off")

    ax_plan.clear()
    ax_plan.set_title(f"Latent Planning (step {step}/{horizon})", fontsize=11)
    ax_plan.set_xlabel("PC1")
    ax_plan.set_ylabel("PC2")

    # Plot ground truth trajectory
    ax_plan.plot(gt_2d[:frame_idx+1, 0], gt_2d[:frame_idx+1, 1],
                 'k-', linewidth=2.5, alpha=0.8, label="Ground Truth", zorder=10)
    ax_plan.plot(gt_2d[frame_idx, 0], gt_2d[frame_idx, 1],
                 'k*', markersize=15, zorder=11)

    # Plot sample trajectories up to current step
    t = min(step + 3, pred_2d.shape[1] - 1)
    for s in range(0, num_samples, 4):  # show every 4th
        alpha = 0.05
        color = '#BBDEFB'
        if s == best_idx:
            alpha = 0.9
            color = '#4CAF50'
        elif s == worst_idx:
            alpha = 0.5
            color = '#F44336'
        ax_plan.plot(pred_2d[s, :t+1, 0], pred_2d[s, :t+1, 1],
                     '-', color=color, alpha=alpha, linewidth=1)

    # Highlight best
    ax_plan.plot(pred_2d[best_idx, :t+1, 0], pred_2d[best_idx, :t+1, 1],
                 '-', color='#4CAF50', linewidth=2.5, alpha=0.9, label="Best Plan", zorder=9)
    ax_plan.plot(pred_2d[worst_idx, :t+1, 0], pred_2d[worst_idx, :t+1, 1],
                 '-', color='#F44336', linewidth=2, alpha=0.7, label="Worst Plan", zorder=8)

    ax_plan.legend(fontsize=8, loc='upper left')

    # Cost distribution
    ax_cost.clear()
    ax_cost.hist(costs, bins=30, color='#90CAF9', edgecolor='#1565C0', alpha=0.8)
    ax_cost.axvline(costs[best_idx], color='#4CAF50', linewidth=2, label=f"Best: {costs[best_idx]:.4f}")
    ax_cost.axvline(costs[worst_idx], color='#F44336', linewidth=2, label=f"Worst: {costs[worst_idx]:.4f}")
    ax_cost.set_title("Planning Cost Distribution", fontsize=11)
    ax_cost.set_xlabel("MSE to Goal")
    ax_cost.set_ylabel("Count")
    ax_cost.legend(fontsize=8)

    fig2.suptitle(f"LeWorldModel Planning - {num_samples} candidates, {rollout_time:.1f}s rollout",
                  fontsize=12, fontweight='bold')
    return []

print("  Rendering planning animation...")
anim2 = animation.FuncAnimation(fig2, animate_planning, frames=horizon, interval=400, blit=False)
anim2.save("lewm_planning.gif", writer="pillow", fps=3, dpi=100)
plt.close()
print("  Saved: lewm_planning.gif")


# ============================================================
# Animation 3: Embedding evolution / anti-collapse visualization
# ============================================================
print("\n--- Animation 3: Embedding Space (Anti-Collapse) ---")

# Encode many frames from different episodes
sample_indices = np.linspace(0, len(dataset) - 1, 500, dtype=int)
all_frames_emb = []

with torch.no_grad():
    batch_size = 20
    for i in range(0, len(sample_indices), batch_size):
        batch_idx = sample_indices[i:i+batch_size]
        batch_pixels = []
        for idx in batch_idx:
            row = dataset.get_row_data(int(idx))
            batch_pixels.append(img_transform(row["pixels"]))

        pixels_batch = torch.stack(batch_pixels).unsqueeze(0).to(DEVICE)
        dummy_acts = torch.zeros(1, len(batch_idx), FRAMESKIP * ACTION_DIM, device=DEVICE)
        info = {"pixels": pixels_batch, "action": dummy_acts}
        out = model.encode(info)
        all_frames_emb.append(out["emb"][0].cpu().numpy())

all_frames_emb = np.vstack(all_frames_emb)
print(f"  Encoded {len(all_frames_emb)} frames")

# PCA
pca3 = PCA(n_components=2)
emb_2d_all = pca3.fit_transform(all_frames_emb)

# Color by position in dataset (proxy for episode/time)
time_color = np.linspace(0, 1, len(emb_2d_all))

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

def animate_embedding(frame):
    n = min((frame + 1) * 20, len(emb_2d_all))

    axes3[0].clear()
    scatter = axes3[0].scatter(emb_2d_all[:n, 0], emb_2d_all[:n, 1],
                                c=time_color[:n], cmap='viridis', s=15, alpha=0.6)
    axes3[0].set_title(f"Embedding Space ({n}/{len(emb_2d_all)} frames)", fontsize=11)
    axes3[0].set_xlabel("PC1")
    axes3[0].set_ylabel("PC2")
    axes3[0].set_xlim(emb_2d_all[:, 0].min() - 1, emb_2d_all[:, 0].max() + 1)
    axes3[0].set_ylim(emb_2d_all[:, 1].min() - 1, emb_2d_all[:, 1].max() + 1)

    # Embedding norm distribution
    axes3[1].clear()
    norms = np.linalg.norm(all_frames_emb[:n], axis=-1)
    axes3[1].hist(norms, bins=40, color='#7E57C2', alpha=0.7, edgecolor='#4527A0')
    axes3[1].set_title("Embedding Norm Distribution\n(SIGReg prevents collapse)", fontsize=11)
    axes3[1].set_xlabel("L2 Norm")
    axes3[1].set_ylabel("Count")
    axes3[1].axvline(norms.mean(), color='red', linewidth=2, linestyle='--',
                     label=f"Mean: {norms.mean():.1f}")
    axes3[1].legend(fontsize=9)

    fig3.suptitle("LeWorldModel - Learned Latent Space (no collapse!)",
                  fontsize=12, fontweight='bold')
    return []

n_anim_frames = len(emb_2d_all) // 20
print(f"  Rendering embedding animation ({n_anim_frames} frames)...")
anim3 = animation.FuncAnimation(fig3, animate_embedding, frames=n_anim_frames, interval=100, blit=False)
anim3.save("lewm_embeddings.gif", writer="pillow", fps=10, dpi=100)
plt.close()
print("  Saved: lewm_embeddings.gif")

print("\n" + "="*50)
print("All animations saved!")
print("  1. lewm_latent_trajectory.gif  - Latent trajectory as agent moves")
print("  2. lewm_planning.gif           - Planning with action candidates")
print("  3. lewm_embeddings.gif         - Embedding space (anti-collapse)")
print("="*50)
