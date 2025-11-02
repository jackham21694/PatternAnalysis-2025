"""
Shows example usage of the trained model. Prints out results and provides visualisations.

"""

import tensorflow as tf
from dataset import data_visualisation
from utils import denormalise, batch_generator
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
#-------------------------------------------------------------PREDICT.PY-----------------


def reconstructed_indicators(data, autoencoder_model, price_min, price_max):
    mse_loss = tf.keras.losses.MeanSquaredError()

    X_reconstructed = autoencoder_model(data).numpy()

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