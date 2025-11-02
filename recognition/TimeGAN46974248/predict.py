"""
Module: predict.py
Author: Jack Ham, 46974248

Description
-----------

This modules contains all of the prediction/visualiation code, 
that shows the performance and results of our autoencoder and
generator. The evaluate_autoencoder function and the 
generate_lob_heatmap_report are both AI developed visualisation
functions.

"""

import tensorflow as tf
from dataset import data_visualisation
from utils import denormalise, batch_generator, random_generator
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
from skimage.metrics import structural_similarity as ssim
from IPython.display import Image, display

#-------------------------------------------------------------PREDICT.PY-----------------


def reconstructed_indicators(data, model, price_min, price_max):
    mse_loss = tf.keras.losses.MeanSquaredError()

    X_reconstructed = model(data).numpy()

    X_denorm = denormalise(data, price_min, price_max)
    X__recon_denorm = denormalise(X_reconstructed, price_min, price_max)

    mid_orig, mid_recon, spread_orig, spread_recon, return_orig, return_recon = data_visualisation(
        X_denorm, X__recon_denorm
    )

    mid_loss = mse_loss(mid_orig, mid_recon)
    spread_loss = mse_loss(spread_orig, spread_recon)
    return_loss = mse_loss(return_orig, return_recon)
    print("MidPrice loss on training set:", mid_loss.numpy())
    print("Spread loss on training set:", spread_loss.numpy())
    print("Return loss on training set:", return_loss.numpy())




def evaluate_autoencoder(model, X_test, seq_len=100, batch_size=128):
    # Get a test batch
    X_mb, _ = batch_generator(X_test, [seq_len]*len(X_test), batch_size)
    X_mb = tf.convert_to_tensor(np.array(X_mb, dtype=np.float32))

    # Get reconstruction
    X_recon = model(X_mb, training=False).numpy()
    X_mb = X_mb.numpy()

    print("=" * 70)
    print("AUTOENCODER EVALUATION (PRICE-ONLY, DEPTH 10)")
    print("=" * 70)

    # 1. Overall MSE (normalized)
    mse = np.mean((X_mb - X_recon) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(X_mb - X_recon))

    print(f"\n✓ Data is normalized: [{X_mb.min():.3f}, {X_mb.max():.3f}]")
    print(f"\n1. RECONSTRUCTION ERROR:")
    print(f"   MSE:  {mse:.6f}")
    print(f"   RMSE: {rmse:.6f}  {'✓ GOOD' if rmse < 0.05 else '✗ NEEDS WORK'}")
    print(f"   MAE:  {mae:.6f}  {'✓ GOOD' if mae < 0.05 else '✗ NEEDS WORK'}")
    print(f"   10*sqrt(MSE): {10*rmse:.4f}  (paper's loss scale)")

    # 2. Correlation
    corr, _ = pearsonr(X_mb.flatten(), X_recon.flatten())
    print(f"\n2. CORRELATION:")
    print(f"   Pearson: {corr:.4f}  {'✓ GOOD' if corr > 0.95 else '✗ NEEDS WORK'}")

    # 3. Per-feature errors (20 price features: 10 ask + 10 bid)
    feature_mse = np.mean((X_mb - X_recon) ** 2, axis=(0,1))
    ask_mse = feature_mse[:10].mean()
    bid_mse = feature_mse[10:].mean()
    price_mse = feature_mse.mean()

    print(f"\n3. PER-FEATURE MSE:")
    print(f"   All Prices: {price_mse:.6f}  (RMSE: {np.sqrt(price_mse):.6f})")
    print(f"   Ask Prices: {ask_mse:.6f}  (RMSE: {np.sqrt(ask_mse):.6f})")
    print(f"   Bid Prices: {bid_mse:.6f}  (RMSE: {np.sqrt(bid_mse):.6f})")

    # 4. VARIANCE PRESERVATION (detects mean collapse!)
    var_real = np.var(X_mb, axis=(0,1))
    var_recon = np.var(X_recon, axis=(0,1))
    var_ratio = var_recon / (var_real + 1e-8)

    avg_var_ratio = np.mean(var_ratio)
    ask_var_ratio = np.mean(var_ratio[:10])
    bid_var_ratio = np.mean(var_ratio[10:])

    print(f"\n4. VARIANCE PRESERVATION (CRITICAL!):")
    print(f"   Overall:    {avg_var_ratio:.3f}  {'✓ GOOD' if avg_var_ratio > 0.7 else '✗ COLLAPSED'}")
    print(f"   Ask Prices: {ask_var_ratio:.3f}  {'✓' if ask_var_ratio > 0.7 else '✗ MEAN COLLAPSE'}")
    print(f"   Bid Prices: {bid_var_ratio:.3f}  {'✓' if bid_var_ratio > 0.7 else '✗ MEAN COLLAPSE'}")

    # 5. Latent space check
    H = model.embed(X_mb)
    H_std = np.std(H.numpy(), axis=(0,1))
    collapsed = np.sum(H_std < 0.01)

    print(f"\n5. LATENT SPACE:")
    print(f"   Collapsed dims: {collapsed}/{len(H_std)}  {'✓ HEALTHY' if collapsed < len(H_std)*0.1 else '✗ COLLAPSED'}")
    print(f"   Avg std: {H_std.mean():.4f}")

    # 6. Overall grade with variance check
    print("\n" + "=" * 70)
    score = 0

    # RMSE scoring (what matters with sqrt loss)
    if rmse < 0.03: score += 2
    elif rmse < 0.05: score += 1

    if corr > 0.97: score += 1
    elif corr > 0.95: score += 0.5

    # CRITICAL: Variance preservation
    if avg_var_ratio > 0.8: score += 3
    elif avg_var_ratio > 0.7: score += 2
    elif avg_var_ratio > 0.5: score += 1

    if collapsed < len(H_std) * 0.1: score += 1

    if score >= 6:
        print("GRADE: ✓✓✓ EXCELLENT - Ready for TimeGAN training!")
    elif score >= 4:
        print("GRADE: ✓✓ GOOD - Should work for TimeGAN")
    elif score >= 2:
        print("GRADE: ✓ ACCEPTABLE - May struggle with variance")
        print("⚠️  Variance ratio < 0.7 suggests mean collapse")
    else:
        print("GRADE: ✗ POOR - MEAN COLLAPSE DETECTED!")
        print("\n⚠️  PROBLEM: Model predicting mean instead of learning patterns")
        print("FIX: Use 10*sqrt(MSE) loss from original paper")
    print("=" * 70)

    # 7. Detailed visualization
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    sample_idx = 0

    # Plot sample reconstruction - Ask and Bid L1
    axes[0,0].plot(X_mb[sample_idx,:,0], label='Real', linewidth=2)
    axes[0,0].plot(X_recon[sample_idx,:,0], label='Recon', linewidth=2, alpha=0.7)
    axes[0,0].set_title(f'Ask Price L1 (var ratio={var_ratio[0]:.2f})')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(X_mb[sample_idx,:,10], label='Real', linewidth=2)
    axes[0,1].plot(X_recon[sample_idx,:,10], label='Recon', linewidth=2, alpha=0.7)
    axes[0,1].set_title(f'Bid Price L1 (var ratio={var_ratio[10]:.2f})')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Per-feature RMSE (20 price features)
    feature_rmse = np.sqrt(feature_mse)
    colors = ['blue']*10 + ['orange']*10  # Ask prices blue, Bid prices orange
    axes[1,0].bar(range(20), feature_rmse, color=colors)
    axes[1,0].axhline(0.05, color='r', linestyle='--', label='Target')
    axes[1,0].axvline(9.5, color='gray', linestyle=':', alpha=0.5)
    axes[1,0].set_xlabel('Feature (blue=ask, orange=bid)')
    axes[1,0].set_ylabel('RMSE')
    axes[1,0].set_title('Per-Feature Error')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Variance ratio per feature (20 price features)
    axes[1,1].bar(range(20), var_ratio, color=colors)
    axes[1,1].axhline(1.0, color='g', linestyle='--', label='Perfect')
    axes[1,1].axhline(0.7, color='r', linestyle='--', label='Acceptable')
    axes[1,1].axvline(9.5, color='gray', linestyle=':', alpha=0.5)
    axes[1,1].set_xlabel('Feature (blue=ask, orange=bid)')
    axes[1,1].set_ylabel('Variance Ratio')
    axes[1,1].set_title('Variance Preservation (CRITICAL)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    # Distribution comparison - Ask Price L1
    axes[2,0].hist(X_mb[:,:,0].flatten(), bins=50, alpha=0.5, label='Real', density=True)
    axes[2,0].hist(X_recon[:,:,0].flatten(), bins=50, alpha=0.5, label='Recon', density=True)
    axes[2,0].set_xlabel('Value')
    axes[2,0].set_ylabel('Density')
    axes[2,0].set_title('Ask Price L1 Distribution')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)

    # Distribution comparison - Bid Price L1
    axes[2,1].hist(X_mb[:,:,10].flatten(), bins=50, alpha=0.5, label='Real', density=True)
    axes[2,1].hist(X_recon[:,:,10].flatten(), bins=50, alpha=0.5, label='Recon', density=True)
    axes[2,1].set_xlabel('Value')
    axes[2,1].set_ylabel('Density')
    axes[2,1].set_title('Bid Price L1 Distribution')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/autoencoder_eval.png', dpi=150)
    plt.show()

    return mse, corr, avg_var_ratio


def generate_lob_heatmap_report(
    generator, supervisor, autoencoder,
    X_test,
    hidden_dim=64, seq_len=100,
    num_samples=5, save_dir='/content/heatmaps'
):
    """
    Generates N (default 5) real vs synthetic LOB heatmaps with SSIM.
    Saves PNGs + displays inline in Colab.
    
    Args:
        generator, supervisor, autoencoder: Trained TimeGAN models
        X_test: numpy array [n_test, seq_len, 20]
        random_generator: your noise function
        hidden_dim, seq_len: model hyperparameters
        num_samples: how many pairs to generate
        save_dir: where to save PNGs
    """
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Helper: one synthetic snapshot
    # ------------------------------------------------------------------
    def one_synthetic():
        Z = random_generator(1, hidden_dim, [seq_len], seq_len)
        H = generator(Z, training=False)
        S = supervisor(H, training=False)
        X = autoencoder.recovery(S)
        return X[0, 0, :].numpy()  # [20]

    # ------------------------------------------------------------------
    # 2. Heatmap from [20] → [10,2]
    # ------------------------------------------------------------------
    def to_heatmap(snap, relative=True):
        ask, bid = snap[0:10], snap[10:20]
        mid = (ask[0] + bid[0]) / 2.0
        if relative:
            bid = (bid - mid) / (mid + 1e-8)
            ask = (ask - mid) / (mid + 1e-8)
        return np.column_stack([bid[::-1], ask])  # best bid on top

    # ------------------------------------------------------------------
    # 3. SSIM with padding (10×2 → 10×7)
    # ------------------------------------------------------------------
    def calc_ssim(real_hm, synth_hm):
        pad_r = np.pad(real_hm,  ((0,0), (0,5)), mode='reflect')
        pad_s = np.pad(synth_hm, ((0,0), (0,5)), mode='reflect')
        dr = pad_r.max() - pad_r.min() + 1e-8
        return ssim(pad_r, pad_s, data_range=dr, win_size=7)

    # ------------------------------------------------------------------
    # 4. Plot + save + display one pair
    # ------------------------------------------------------------------
    def plot_pair(real_snap, synth_snap, idx):
        real_hm  = to_heatmap(real_snap)
        synth_hm = to_heatmap(synth_snap)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        im1 = ax1.imshow(real_hm,  cmap='viridis', aspect='auto')
        ax1.set_title(f'Real LOB #{idx}'); ax1.set_xlabel('Side'); ax1.set_ylabel('Level')
        plt.colorbar(im1, ax=ax1, label='Rel. price')

        im2 = ax2.imshow(synth_hm, cmap='viridis', aspect='auto')
        ax2.set_title(f'Synthetic LOB #{idx}'); ax2.set_xlabel('Side')
        plt.colorbar(im2, ax=ax2, label='Rel. price')

        s = calc_ssim(real_hm, synth_hm)
        fig.suptitle(f'SSIM = {s:.4f}')
        plt.tight_layout()

        path = f'{save_dir}/lob_snapshot_{idx}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return s, path

    # ------------------------------------------------------------------
    # 5. Main loop
    # ------------------------------------------------------------------
    real_flat = X_test.reshape(-1, 20)
    indices = np.random.choice(len(real_flat), size=num_samples, replace=False)

    ssim_list = []
    for i, pos in enumerate(indices, 1):
        real_snap  = real_flat[pos]
        synth_snap = one_synthetic()
        ssim_val, png_path = plot_pair(real_snap, synth_snap, i)
        ssim_list.append(ssim_val)
        print(f'Pair {i}: SSIM = {ssim_val:.4f}')
        display(Image(png_path))

    avg_ssim = np.mean(ssim_list)
    print(f'\nAverage SSIM: {avg_ssim:.4f}')
    return avg_ssim, ssim_list