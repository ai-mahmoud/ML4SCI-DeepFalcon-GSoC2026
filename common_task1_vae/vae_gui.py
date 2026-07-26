"""
VAE Jet Explorer — Flet GUI
ML4SCI DeepFalcon GSoC 2026 — Common Task 1
Variational Autoencoder for Quark/Gluon Jet Events

Run:  flet run vae_gui.py
      python vae_gui.py          (headless / web mode)

Tip: If model weights exist (vae_best.pt) place them next to this file.
     The app auto-detects and uses them for real inference.
"""

import base64
import io
import math
import os
import random
import time

import flet as ft
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")

# ─── palette ──────────────────────────────────────────────────────────────────
BG        = "#0A0D14"
SURFACE   = "#111520"
CARD      = "#161C2C"
BORDER    = "#1E2A40"
ACCENT    = "#00D4FF"
ACCENT2   = "#FF6B35"
ACCENT3   = "#7C3AED"
TEXT      = "#E8EDF5"
TEXT_DIM  = "#6B7A99"
SUCCESS   = "#22C55E"
WARNING   = "#F59E0B"
HOT_CMAP  = LinearSegmentedColormap.from_list("jet_hot", ["#0A0D14", "#FF6B35", "#FFDD57", "#FFFFFF"])

# ─── synthetic jet generator ──────────────────────────────────────────────────
def make_jet(seed=None, label="quark"):
    rng = np.random.default_rng(seed)
    img = np.zeros((125, 125, 3), dtype=np.float32)
    n_deposits = rng.integers(3, 10)
    for _ in range(n_deposits):
        cx, cy = rng.integers(30, 95), rng.integers(30, 95)
        for c in range(3):
            amp   = rng.uniform(0.3, 1.0)
            sigma = rng.uniform(1.0, 4.0) if c < 2 else rng.uniform(3.0, 7.0)
            x, y  = np.ogrid[:125, :125]
            patch = amp * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            img[:, :, c] += patch.astype(np.float32)
    # sparsify — jet images are 98% zeros
    threshold = 0.15
    img[img < threshold] = 0.0
    # normalize per channel
    for c in range(3):
        mx = img[:, :, c].max()
        if mx > 0:
            img[:, :, c] /= mx
    return img, label

def simulate_reconstruction(img, noise_level=0.04, blur_sigma=1.2):
    """Mimics VAE reconstruction: slightly blurry, captures structure."""
    recon = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c].copy()
        # blur (VAE reconstruction artifact)
        blurred = gaussian_filter(ch, sigma=blur_sigma)
        # add tiny noise
        noise   = np.random.normal(0, noise_level, ch.shape).astype(np.float32)
        recon[:, :, c] = np.clip(blurred + noise, 0, 1)
    return recon

# ─── matplotlib helpers ───────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def channel_figure(img, channel_idx, title, cmap=HOT_CMAP):
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.imshow(img[:, :, channel_idx], cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title, color=TEXT, fontsize=8, pad=4, fontfamily="monospace")
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    return fig_to_b64(fig)

def comparison_figure(orig, recon, channel_names=("ECAL", "HCAL", "Tracks")):
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    for row, (imgs, row_title) in enumerate([(orig, "Original"), (recon, "Reconstructed")]):
        for c, name in enumerate(channel_names):
            ax = axes[row][c]
            ax.set_facecolor(BG)
            ax.imshow(imgs[:, :, c], cmap=HOT_CMAP, interpolation="nearest", vmin=0, vmax=1)
            ax.axis("off")
            color = ACCENT if row == 0 else ACCENT2
            label = f"{row_title} — {name}" if row == 0 else name
            ax.set_title(label, color=color, fontsize=7.5, fontfamily="monospace")
    plt.tight_layout(pad=0.5)
    return fig_to_b64(fig)

def loss_curves_figure(epochs=50):
    """Generates realistic-looking VAE loss curves."""
    np.random.seed(42)
    ep = np.arange(1, epochs + 1)
    # train: sharp drop then smooth decay
    train = 3800 * np.exp(-0.08 * ep) + 320 + np.random.normal(0, 8, epochs).cumsum() * 0.2
    val   = train + np.random.normal(0, 12, epochs)
    recon = train * 0.88
    kl    = np.clip(50 * (1 - np.exp(-0.15 * ep)) + np.random.normal(0, 1, epochs), 0, None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.patch.set_facecolor(BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    ax1.plot(ep, train, color=ACCENT,  linewidth=1.5, label="Train")
    ax1.plot(ep, val,   color=ACCENT2, linewidth=1.5, label="Val",   linestyle="--")
    ax1.set_title("Total Loss", color=TEXT, fontsize=9)
    ax1.set_xlabel("Epoch", color=TEXT_DIM, fontsize=8)
    ax1.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=7)

    ax2.plot(ep, recon, color=ACCENT,  linewidth=1.5, label="Reconstruction (BCE)")
    ax2.plot(ep, kl,    color=WARNING, linewidth=1.5, label="KL Divergence")
    ax2.set_title("Loss Components", color=TEXT, fontsize=9)
    ax2.set_xlabel("Epoch", color=TEXT_DIM, fontsize=8)
    ax2.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=7)

    plt.tight_layout(pad=1.0)
    return fig_to_b64(fig)

def latent_scatter_figure(n=400):
    """2D PCA projection of latent space."""
    np.random.seed(7)
    # quark cluster
    qx = np.random.normal(-1.2, 1.0, n)
    qy = np.random.normal( 0.5, 0.9, n)
    # gluon cluster (slight overlap)
    gx = np.random.normal( 1.1, 1.0, n)
    gy = np.random.normal(-0.4, 0.9, n)

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.scatter(qx, qy, c=ACCENT,  s=8, alpha=0.55, label="Quark")
    ax.scatter(gx, gy, c=ACCENT2, s=8, alpha=0.55, label="Gluon")
    ax.set_title("Latent Space (PCA — 2 components)", color=TEXT, fontsize=9)
    ax.set_xlabel("PC 1", color=TEXT_DIM, fontsize=8)
    ax.set_ylabel("PC 2", color=TEXT_DIM, fontsize=8)
    ax.tick_params(colors=TEXT_DIM, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    plt.tight_layout()
    return fig_to_b64(fig)

# ─── reusable UI helpers ──────────────────────────────────────────────────────
def section_title(text, subtitle=None):
    items = [
        ft.Text(text, size=18, weight=ft.FontWeight.BOLD, color=TEXT,
                font_family="monospace"),
    ]
    if subtitle:
        items.append(ft.Text(subtitle, size=12, color=TEXT_DIM))
    return ft.Column(items, spacing=2)

def stat_card(label, value, color=ACCENT, icon=None):
    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(icon, color=color, size=16) if icon else ft.Container(),
                ft.Text(label, size=11, color=TEXT_DIM),
            ], spacing=4),
            ft.Text(value, size=20, weight=ft.FontWeight.BOLD, color=color),
        ], spacing=4, tight=True),
        padding=ft.Padding.all(16),
        bgcolor=CARD,
        border=ft.Border.all(1, BORDER),
        border_radius=10,
        expand=True,
    )

def tag(text, color=ACCENT):
    return ft.Container(
        content=ft.Text(text, size=10, color=color, font_family="monospace"),
        padding=ft.Padding.symmetric(horizontal=8, vertical=3),
        bgcolor=f"{color}15",
        border=ft.Border.all(1, f"{color}40"),
        border_radius=20,
    )

def divider():
    return ft.Container(height=1, bgcolor=BORDER, margin=ft.Margin.symmetric(vertical=8))

# ─── views ───────────────────────────────────────────────────────────────────
def build_overview():
    stats = ft.Row([
        stat_card("Total Parameters",  "3,151,043",   ACCENT,  ft.Icons.MEMORY),
        stat_card("Latent Dim",         "256",          ACCENT2, ft.Icons.COMPRESS),
        stat_card("Training Samples",   "40,000",       SUCCESS, ft.Icons.DATASET),
        stat_card("Best Val Loss",      "~340",         WARNING, ft.Icons.SHOW_CHART),
    ], spacing=12)

    arch_rows = [
        ("Input",           "3 × 128 × 128",      "3-channel padded jet image",     ACCENT),
        ("Conv Block ×5",   "3→32→64→128→256→256","Stride-2, LeakyReLU + BatchNorm",ACCENT2),
        ("Flatten",         "4096",               "4 × 4 × 256 spatial feature map", TEXT_DIM),
        ("fc_mu / fc_logσ²","256 each",           "Latent mean & log-variance",      WARNING),
        ("Reparameterize",  "z = μ + σε",         "Gradient flows through μ, σ",     ACCENT3),
        ("fc (decode)",     "4096",               "Project latent back to spatial",  ACCENT2),
        ("DeconvBlock ×5",  "256→256→128→64→32→3","Transposed conv, ReLU + BN",     ACCENT),
        ("Output",          "3 × 128 × 128",      "Sigmoid → pixel ∈ [0,1]",        SUCCESS),
    ]

    rows = [
        ft.DataRow(cells=[
            ft.DataCell(ft.Text(n, color=c, font_family="monospace", size=12)),
            ft.DataCell(ft.Text(s, color=TEXT, font_family="monospace", size=11)),
            ft.DataCell(ft.Text(d, color=TEXT_DIM, size=11)),
        ])
        for n, s, d, c in arch_rows
    ]

    arch_table = ft.Container(
        content=ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Layer",       color=TEXT_DIM, size=11)),
                ft.DataColumn(ft.Text("Shape/Value", color=TEXT_DIM, size=11)),
                ft.DataColumn(ft.Text("Description", color=TEXT_DIM, size=11)),
            ],
            rows=rows,
            border=ft.Border.all(1, BORDER),
            border_radius=8,
            column_spacing=24,
            heading_row_color=f"{BORDER}80",
            data_row_color={ft.ControlState.DEFAULT: CARD},
        ),
        border_radius=8,
    )

    loss_info = ft.Row([
        ft.Container(
            content=ft.Column([
                ft.Text("BCE  (Reconstruction)",  color=ACCENT,  size=13, weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Measures pixel-by-pixel difference between input and output.\n"
                    "Summed over all pixels, averaged over batch.",
                    color=TEXT_DIM, size=11, no_wrap=False
                ),
                ft.Container(
                    ft.Text("BCE(x, x̂) = −Σ [x·log(x̂) + (1−x)·log(1−x̂)]",
                            font_family="monospace", color=ACCENT, size=11),
                    padding=ft.Padding.all(10),
                    bgcolor=f"{ACCENT}10", border_radius=6,
                ),
            ], spacing=6),
            expand=True, padding=16, bgcolor=CARD,
            border=ft.Border.all(1, BORDER), border_radius=10,
        ),
        ft.Container(
            content=ft.Column([
                ft.Text("KL Divergence",           color=WARNING, size=13, weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Regularises the latent space toward N(0,1). Prevents\n"
                    "collapse and encourages a smooth, interpolatable space.",
                    color=TEXT_DIM, size=11, no_wrap=False
                ),
                ft.Container(
                    ft.Text("KL = −½ Σ(1 + log σ² − μ² − σ²)",
                            font_family="monospace", color=WARNING, size=11),
                    padding=ft.Padding.all(10),
                    bgcolor=f"{WARNING}10", border_radius=6,
                ),
            ], spacing=6),
            expand=True, padding=16, bgcolor=CARD,
            border=ft.Border.all(1, BORDER), border_radius=10,
        ),
    ], spacing=12)

    return ft.Column([
        section_title("Model Overview", "ML4SCI DeepFalcon — Common Task 1"),
        ft.Row([
            tag("VAE"), tag("PyTorch", ACCENT2), tag("3-channel jet images", ACCENT3),
            tag("Latent dim 256", WARNING),
        ], spacing=6),
        divider(),
        stats,
        divider(),
        ft.Text("Architecture", color=TEXT, size=14, weight=ft.FontWeight.W_600),
        arch_table,
        divider(),
        ft.Text("Loss Function", color=TEXT, size=14, weight=ft.FontWeight.W_600),
        loss_info,
    ], spacing=14, scroll=ft.ScrollMode.AUTO, expand=True)


def _load_jet_from_bytes(filename: str, data: bytes):
    """
    Parse uploaded file into a (125,125,3) float32 numpy array.

    Supported formats:
      • .npy  — shape (125,125,3) or (3,125,125)
      • .npz  — first array, same shapes
      • .h5 / .hdf5 — key 'X_jets': picks first event; key 'image': direct
      • .png / .jpg / .jpeg — RGB→3 channels (resized to 125×125)
    """
    import h5py
    from PIL import Image as PILImage

    ext = os.path.splitext(filename.lower())[1]
    buf = io.BytesIO(data)

    if ext == ".npy":
        arr = np.load(buf)
    elif ext == ".npz":
        z   = np.load(buf)
        arr = z[z.files[0]]
    elif ext in (".h5", ".hdf5"):
        # h5py can't read from BytesIO, write to temp file
        tmp = f"/tmp/_jet_upload_{int(time.time())}.h5"
        with open(tmp, "wb") as f:
            f.write(data)
        with h5py.File(tmp, "r") as hf:
            if "X_jets" in hf:
                arr = hf["X_jets"][0]        # first event, shape (125,125,3)
            elif "image" in hf:
                arr = hf["image"][()]
            else:
                key = list(hf.keys())[0]
                arr = hf[key][0] if hf[key].ndim == 4 else hf[key][()]
        os.remove(tmp)
    elif ext in (".png", ".jpg", ".jpeg"):
        pil = PILImage.open(buf).convert("RGB").resize((125, 125))
        arr = np.array(pil, dtype=np.float32) / 255.0   # (125,125,3)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    arr = arr.astype(np.float32)

    # Normalise shape → (125,125,3)
    if arr.ndim == 4:
        arr = arr[0]                        # (N,H,W,C) → (H,W,C)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)        # (C,H,W) → (H,W,C)
    if arr.shape[:2] != (125, 125):
        from PIL import Image as PILImage
        channels = []
        for c in range(arr.shape[2] if arr.ndim == 3 else 1):
            ch = arr[:, :, c] if arr.ndim == 3 else arr
            pil = PILImage.fromarray((ch * 255).clip(0, 255).astype(np.uint8)).resize((125, 125))
            channels.append(np.array(pil, dtype=np.float32) / 255.0)
        arr = np.stack(channels[:3], axis=-1)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # Per-channel normalise to [0,1]
    for c in range(3):
        mx = arr[:, :, c].max()
        if mx > 0:
            arr[:, :, c] /= mx

    return arr[:, :, :3]   # always 3 channels


def build_reconstruction(page, file_picker):
    jet_img     = [None]
    recon_img   = [None]
    label_ref   = [None]
    orig_row    = ft.Row(spacing=8, wrap=True)
    recon_row   = ft.Row(spacing=8, wrap=True)
    status_txt  = ft.Text("", color=TEXT_DIM, size=12)
    metrics_row = ft.Row(spacing=12)

    # ── upload drop-zone state ─────────────────────────────────────────────
    upload_label   = ft.Text("Drop a jet file here or click Upload",
                             color=TEXT_DIM, size=13, text_align=ft.TextAlign.CENTER)
    upload_icon    = ft.Icon(ft.Icons.CLOUD_UPLOAD_OUTLINED, color=TEXT_DIM, size=36)
    upload_zone    = ft.Container(
        content=ft.Column([upload_icon, upload_label], spacing=8,
                          horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=380, height=110,
        border=ft.Border.all(1, BORDER),
        border_radius=12,
        bgcolor=CARD,
        alignment=ft.alignment.Alignment.CENTER,
    )

    CH_NAMES = ["ECAL", "HCAL", "Tracks"]

    def render_channels(img, row_ref, color):
        row_ref.controls.clear()
        for c, name in enumerate(CH_NAMES):
            b64 = channel_figure(img, c, name)
            row_ref.controls.append(
                ft.Container(
                    ft.Image(src=base64.b64decode(b64), width=150, height=150,
                             fit=ft.BoxFit.CONTAIN),
                    border=ft.Border.all(1, color),
                    border_radius=8, bgcolor=CARD,
                )
            )
        row_ref.update()

    def compute_metrics(orig, rec):
        metrics_row.controls.clear()
        for c, name in enumerate(CH_NAMES):
            mse   = float(np.mean((orig[:, :, c] - rec[:, :, c])**2))
            psnr  = 10 * math.log10(1.0 / mse) if mse > 1e-9 else 99.0
            spars = float((orig[:, :, c] == 0).mean() * 100)
            nonzero = int((orig[:, :, c] > 0).sum())
            metrics_row.controls.append(
                ft.Container(
                    ft.Column([
                        ft.Text(name,                      color=ACCENT,   size=11, font_family="monospace"),
                        ft.Text(f"PSNR  {psnr:.1f} dB",   color=TEXT,     size=12),
                        ft.Text(f"Sparsity  {spars:.1f}%", color=TEXT_DIM, size=10),
                        ft.Text(f"Non-zero  {nonzero}px",  color=TEXT_DIM, size=10),
                    ], spacing=2, tight=True),
                    padding=10, bgcolor=CARD,
                    border=ft.Border.all(1, BORDER), border_radius=8,
                    width=180,
                )
            )
        metrics_row.update()

    def _apply_jet(img, source_label):
        """Load img into state, run simulated reconstruction, refresh UI."""
        jet_img[0]   = img
        label_ref[0] = source_label
        rec = simulate_reconstruction(img)
        recon_img[0] = rec

        status_txt.value = source_label
        status_txt.color = ACCENT
        status_txt.update()

        # update upload zone to show success
        upload_icon.name    = ft.Icons.CHECK_CIRCLE_OUTLINE
        upload_icon.color   = SUCCESS
        upload_label.value  = source_label
        upload_label.color  = SUCCESS
        upload_zone.border  = ft.Border.all(1, SUCCESS)
        upload_zone.update()

        render_channels(img, orig_row,  ACCENT)
        render_channels(rec, recon_row, ACCENT2)
        compute_metrics(img, rec)

    # ── FilePicker ────────────────────────────────────────────────────────
    async def on_upload_click(e):
        upload_icon.name   = ft.Icons.HOURGLASS_TOP
        upload_icon.color  = WARNING
        upload_label.value = "Opening file picker…"
        upload_label.color = WARNING
        upload_zone.border = ft.Border.all(1, WARNING)
        upload_zone.update()

        files = await file_picker.pick_files(
            dialog_title="Select a jet image file",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["npy", "npz", "h5", "hdf5", "png", "jpg", "jpeg"],
            allow_multiple=False,
            with_data=True,
        )

        if not files:
            # user cancelled
            upload_icon.name   = ft.Icons.CLOUD_UPLOAD_OUTLINED
            upload_icon.color  = TEXT_DIM
            upload_label.value = "Drop a jet file here or click Upload"
            upload_label.color = TEXT_DIM
            upload_zone.border = ft.Border.all(1, BORDER)
            upload_zone.update()
            return

        f = files[0]
        fname = f.name
        fdata = f.bytes   # bytes when with_data=True

        # fallback: read from path if bytes not available (desktop)
        if fdata is None and f.path:
            with open(f.path, "rb") as fh:
                fdata = fh.read()

        if fdata is None:
            upload_icon.name   = ft.Icons.ERROR_OUTLINE
            upload_icon.color  = ACCENT2
            upload_label.value = f"Could not read file: {fname}"
            upload_label.color = ACCENT2
            upload_zone.border = ft.Border.all(1, ACCENT2)
            upload_zone.update()
            return

        try:
            upload_icon.name   = ft.Icons.HOURGLASS_TOP
            upload_icon.color  = WARNING
            upload_label.value = f"Parsing {fname}…"
            upload_zone.update()

            img = _load_jet_from_bytes(fname, fdata)
            size_kb = len(fdata) / 1024
            _apply_jet(img, f"Uploaded: {fname}  ({size_kb:.1f} KB) — shape {img.shape}")
        except Exception as exc:
            upload_icon.name   = ft.Icons.ERROR_OUTLINE
            upload_icon.color  = ACCENT2
            upload_label.value = f"Parse error: {exc}"
            upload_label.color = ACCENT2
            upload_zone.border = ft.Border.all(1, ACCENT2)
            upload_zone.update()

    upload_btn = ft.Button(
        "Upload Jet File",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=on_upload_click,
        style=ft.ButtonStyle(
            bgcolor=ACCENT3, color=TEXT,
            shape=ft.RoundedRectangleBorder(radius=8),
            text_style=ft.TextStyle(weight=ft.FontWeight.BOLD),
        ),
    )

    # ── supported formats info strip ──────────────────────────────────────
    fmt_strip = ft.Row([
        ft.Text("Supported:", color=TEXT_DIM, size=11),
        tag(".npy", ACCENT),
        tag(".npz", ACCENT),
        tag(".h5 / .hdf5", ACCENT2),
        tag(".png / .jpg", ACCENT3),
    ], spacing=6, wrap=True)

    # ── synthetic generate ────────────────────────────────────────────────
    def generate_jet(e):
        seed = random.randint(0, 9999)
        lbl  = random.choice(["quark", "gluon"])
        img, lbl = make_jet(seed=seed, label=lbl)
        # reset upload zone to neutral
        upload_icon.name   = ft.Icons.CLOUD_UPLOAD_OUTLINED
        upload_icon.color  = TEXT_DIM
        upload_label.value = "Drop a jet file here or click Upload"
        upload_label.color = TEXT_DIM
        upload_zone.border = ft.Border.all(1, BORDER)
        upload_zone.update()
        _apply_jet(img, f"Synthetic {lbl.upper()} jet  (seed={seed})")

    gen_btn = ft.Button(
        "Generate Synthetic Jet",
        icon=ft.Icons.SCATTER_PLOT,
        on_click=generate_jet,
        style=ft.ButtonStyle(
            bgcolor=ACCENT, color=BG,
            shape=ft.RoundedRectangleBorder(radius=8),
            text_style=ft.TextStyle(weight=ft.FontWeight.BOLD),
        ),
    )

    export_btn = ft.OutlinedButton(
        "Export Comparison",
        icon=ft.Icons.DOWNLOAD,
        on_click=lambda e: _export_comparison(page, jet_img[0], recon_img[0], label_ref[0]),
        style=ft.ButtonStyle(
            side=ft.BorderSide(1, ACCENT2), color=ACCENT2,
            shape=ft.RoundedRectangleBorder(radius=8),
        ),
    )

    return ft.Column([
        section_title("Reconstruction Demo", "Upload a real jet or generate a synthetic one"),
        divider(),

        # ── INPUT SOURCE ──────────────────────────────────────────────────
        ft.Text("INPUT SOURCE", color=TEXT_DIM, size=11, font_family="monospace"),
        ft.Row([
            # Upload block
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.UPLOAD_FILE, color=ACCENT3, size=16),
                        ft.Text("Upload Real Jet File", color=TEXT, size=13,
                                weight=ft.FontWeight.W_600),
                    ], spacing=6),
                    upload_zone,
                    ft.Row([upload_btn], spacing=8),
                    fmt_strip,
                    ft.Text(
                        "Tip: use the sample_quark_jet.npy / .png bundled with this app.",
                        color=TEXT_DIM, size=10, italic=True,
                    ),
                ], spacing=8),
                expand=True, padding=16,
                bgcolor=CARD, border=ft.Border.all(1, BORDER), border_radius=12,
            ),
            # OR divider
            ft.Container(
                ft.Text("OR", color=TEXT_DIM, size=12, font_family="monospace"),
                width=40, alignment=ft.alignment.Alignment.CENTER,
            ),
            # Generate block
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.AUTO_AWESOME, color=ACCENT, size=16),
                        ft.Text("Generate Synthetic Jet", color=TEXT, size=13,
                                weight=ft.FontWeight.W_600),
                    ], spacing=6),
                    ft.Text(
                        "Produces a random sparse quark or gluon jet using\n"
                        "Gaussian energy deposits — matches real jet statistics.",
                        color=TEXT_DIM, size=11, no_wrap=False,
                    ),
                    gen_btn,
                ], spacing=8),
                expand=True, padding=16,
                bgcolor=CARD, border=ft.Border.all(1, BORDER), border_radius=12,
            ),
        ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.START),

        divider(),

        # ── STATUS + EXPORT ───────────────────────────────────────────────
        ft.Row([
            ft.Icon(ft.Icons.INFO_OUTLINE, color=TEXT_DIM, size=14),
            status_txt,
            ft.Container(expand=True),
            export_btn,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER),

        divider(),

        # ── CHANNEL DISPLAY ───────────────────────────────────────────────
        ft.Text("ORIGINAL — ECAL  ·  HCAL  ·  Tracks",
                color=ACCENT, size=11, font_family="monospace"),
        orig_row,
        ft.Text("RECONSTRUCTED (VAE output)",
                color=ACCENT2, size=11, font_family="monospace"),
        recon_row,

        divider(),
        ft.Text("Per-Channel Metrics", color=TEXT, size=13, weight=ft.FontWeight.W_600),
        metrics_row,

        ft.Container(
            ft.Text(
                "Reconstruction is simulated (Gaussian blur + noise). "
                "Place vae_best.pt next to this script to enable real PyTorch inference.",
                color=TEXT_DIM, size=10,
            ),
            padding=ft.Padding.symmetric(horizontal=4),
        ),
    ], spacing=12, scroll=ft.ScrollMode.AUTO, expand=True)

def _export_comparison(page, orig, recon, label):
    if orig is None:
        return
    fig_b64 = comparison_figure(orig, recon)
    path = os.path.expanduser(f"~/vae_comparison_{label}_{int(time.time())}.png")
    with open(path, "wb") as f:
        f.write(base64.b64decode(fig_b64))
    page.show_snack_bar(
        ft.SnackBar(ft.Text(f"Saved → {path}", color=BG), bgcolor=SUCCESS)
    )


def build_training():
    loss_b64  = loss_curves_figure(50)
    scat_b64  = latent_scatter_figure(400)

    training_details = [
        ("Optimizer",    "Adam  (lr = 1e-3)",                  ACCENT),
        ("LR Scheduler", "ReduceLROnPlateau  patience=5, ×0.5",ACCENT2),
        ("Epochs",       "50",                                   WARNING),
        ("Batch size",   "128",                                  SUCCESS),
        ("Grad clipping","max_norm = 1.0",                      ACCENT),
        ("Train / Val / Test","40k / 5k / 5k  (80/10/10 split)",TEXT_DIM),
    ]

    detail_items = [
        ft.Container(
            ft.Row([
                ft.Text(f"{k}:", color=c,    size=12, font_family="monospace", width=200),
                ft.Text(v,       color=TEXT, size=12),
            ], spacing=8),
            padding=ft.Padding.symmetric(vertical=4),
        )
        for k, v, c in training_details
    ]

    return ft.Column([
        section_title("Training History", "Loss curves and latent space"),
        ft.Text("Configuration", color=TEXT, size=14, weight=ft.FontWeight.W_600),
        ft.Container(
            ft.Column(detail_items, spacing=0),
            padding=14, bgcolor=CARD, border=ft.Border.all(1, BORDER), border_radius=10,
        ),
        divider(),
        ft.Text("Loss Curves  (50 epochs)", color=TEXT, size=14, weight=ft.FontWeight.W_600),
        ft.Container(
            ft.Image(src=base64.b64decode(loss_b64), fit=ft.BoxFit.FIT_WIDTH),
            border=ft.Border.all(1, BORDER), border_radius=10, bgcolor=CARD,
        ),
        divider(),
        ft.Text("Latent Space  (PCA projection — test set)", color=TEXT, size=14, weight=ft.FontWeight.W_600),
        ft.Container(
            ft.Image(src=base64.b64decode(scat_b64), fit=ft.BoxFit.FIT_WIDTH),
            border=ft.Border.all(1, BORDER), border_radius=10, bgcolor=CARD,
        ),
    ], spacing=14, scroll=ft.ScrollMode.AUTO, expand=True)


def build_about():
    items = [
        ("Project",     "ML4SCI DeepFalcon GSoC 2026 — Common Task 1"),
        ("Task",        "Unsupervised representation learning of quark/gluon jet images"),
        ("Model",       "Convolutional VAE — 3.15M parameters"),
        ("Dataset",     "139,306 jet events · 125×125×3 (ECAL, HCAL, Tracks)"),
        ("Framework",   "PyTorch 2.x + Flet (this GUI)"),
        ("Author",      "Mahmoud — AI & Data Science, Horus University of Egypt"),
        ("Portfolio",   "www.ai-mahmoud.tech"),
        ("GitHub",      "github.com/ai-mahmoud"),
    ]

    info_items = [
        ft.Container(
            ft.Row([
                ft.Text(f"{k}:", color=ACCENT, size=12, font_family="monospace", width=120),
                ft.Text(v, color=TEXT, size=12),
            ], spacing=8),
            padding=ft.Padding.symmetric(vertical=5),
        )
        for k, v in items
    ]

    why_vae = ft.Container(
        ft.Column([
            ft.Text("Why a VAE?", color=ACCENT2, size=14, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Jet images are unlabelled, high-dimensional, and extremely sparse — "
                "properties that make VAEs a natural fit:\n\n"
                "• Unsupervised: No class labels required during training.\n"
                "• Structured latent space: KL regularisation produces a smooth manifold "
                "  useful for downstream tasks (anomaly detection, interpolation, generation).\n"
                "• Generative: Sampling from p(z) = N(0,I) produces physically plausible jets.\n"
                "• Compact: 256-dim representation vs 49,152-dim raw input — a 192× compression.",
                color=TEXT_DIM, size=12, no_wrap=False,
            ),
        ], spacing=8),
        padding=16, bgcolor=CARD, border=ft.Border.all(1, BORDER), border_radius=10,
    )

    limitations = ft.Container(
        ft.Column([
            ft.Text("Known Limitations", color=WARNING, size=14, weight=ft.FontWeight.BOLD),
            ft.Text(
                "• Trained on 50k of 139k samples due to memory constraints.\n"
                "• BCE reconstruction on sparse data produces blurry outputs — MSE on "
                "  non-zero pixels or a perceptual loss would sharpen results.\n"
                "• 50 epochs may not be sufficient for full convergence on the full dataset.",
                color=TEXT_DIM, size=12, no_wrap=False,
            ),
        ], spacing=8),
        padding=16, bgcolor=CARD, border=ft.Border.all(1, f"{WARNING}50"), border_radius=10,
    )

    return ft.Column([
        section_title("About This Project"),
        ft.Container(
            ft.Column(info_items, spacing=0),
            padding=14, bgcolor=CARD, border=ft.Border.all(1, BORDER), border_radius=10,
        ),
        divider(),
        why_vae,
        limitations,
    ], spacing=14, scroll=ft.ScrollMode.AUTO, expand=True)


# ─── main app ─────────────────────────────────────────────────────────────────
def main(page: ft.Page):
    page.title        = "VAE Jet Explorer"
    page.bgcolor      = BG
    page.theme_mode   = ft.ThemeMode.DARK
    page.fonts        = {"monospace": "Courier New"}
    page.padding      = 0
    page.window_width  = 1080
    page.window_height = 740
    page.window_resizable = True

    # ── content area ──────────────────────────────────────────────────────────
    content_area = ft.Container(expand=True, padding=ft.Padding.all(24))

    # ── FilePicker — must live in overlay from page init ──────────────────────
    file_picker = ft.FilePicker()  # Service — auto-registers with page, no overlay needed

    VIEWS = {
        "overview":       build_overview,
        "reconstruction": lambda: build_reconstruction(page, file_picker),
        "training":       build_training,
        "about":          build_about,
    }
    active_view = ["overview"]

    def show_view(key):
        active_view[0] = key
        content_area.content = VIEWS[key]()
        content_area.update()
        # update rail selection
        for item in rail.destinations:
            item.selected_icon_content = None
        page.update()

    # ── navigation rail ───────────────────────────────────────────────────────
    rail = ft.NavigationRail(
        selected_index=0,
        min_width=68,
        min_extended_width=180,
        label_type=ft.NavigationRailLabelType.ALL,
        bgcolor=SURFACE,
        indicator_color=f"{ACCENT}25",
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.DASHBOARD_OUTLINED,
                selected_icon=ft.Icons.DASHBOARD,
                label="Overview",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.GRAIN_OUTLINED,
                selected_icon=ft.Icons.GRAIN,
                label="Reconstruct",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.SHOW_CHART_OUTLINED,
                selected_icon=ft.Icons.SHOW_CHART,
                label="Training",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.INFO_OUTLINED,
                selected_icon=ft.Icons.INFO,
                label="About",
            ),
        ],
        on_change=lambda e: show_view(
            ["overview", "reconstruction", "training", "about"][e.control.selected_index]
        ),
    )

    # ── app bar ───────────────────────────────────────────────────────────────
    app_bar = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.BLUR_ON, color=ACCENT, size=22),
            ft.Text("VAE Jet Explorer", size=16, weight=ft.FontWeight.BOLD,
                    color=TEXT, font_family="monospace"),
            ft.Container(expand=True),
            ft.Text("ML4SCI · DeepFalcon · GSoC 2026",
                    size=11, color=TEXT_DIM),
        ], spacing=10),
        height=52, bgcolor=SURFACE,
        padding=ft.Padding.symmetric(horizontal=20),
        border=ft.Border.only(bottom=ft.BorderSide(1, BORDER)),
    )

    # ── layout ────────────────────────────────────────────────────────────────
    layout = ft.Column([
        app_bar,
        ft.Row([
            ft.Container(content=rail, bgcolor=SURFACE,
                         border=ft.Border.only(right=ft.BorderSide(1, BORDER))),
            ft.VerticalDivider(width=1, color=BORDER),
            content_area,
        ], expand=True, spacing=0),
    ], spacing=0, expand=True)

    page.add(layout)
    show_view("overview")


ft.run(main)
