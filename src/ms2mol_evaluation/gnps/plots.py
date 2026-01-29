import re
from pathlib import Path
from typing import Tuple

import dreams.utils.plots as dplots
import dreams.utils.spectra as su
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cycler import cycler
from IPython.display import SVG, display
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.stats import gaussian_kde


def plot_molecule_pair(
    smiles_left: str | None,
    smiles_right: str | None,
    legends=("Query", "Library"),
    size=(300, 300),
    dreams_sim: float | None = None,
    cosine_sim: float | None = None,
    tanimoto: float | None = None,
    header_loc: str = "top-left",  # "top-left" | "top-right"
    save_pth: str | None = None,
):
    """
    Render two SMILES side-by-side and overlay DreaMS / Cosine / Tanimoto text on the SVG.
    """
    left_mol = Chem.MolFromSmiles(smiles_left) if smiles_left else None
    right_mol = Chem.MolFromSmiles(smiles_right) if smiles_right else None

    if not (left_mol and right_mol):
        # Graceful fallback
        if left_mol:
            display(left_mol)
        if right_mol:
            display(right_mol)
        return

    img = Draw.MolsToGridImage(
        [left_mol, right_mol],
        molsPerRow=2,
        subImgSize=size,
        legends=list(legends),
        useSVG=True,
    )

    # Get SVG text
    svg = img.data if hasattr(img, "data") else str(img)

    # Build the metrics text
    parts = []
    if dreams_sim is not None:
        parts.append(f"DreaMS {dreams_sim:.2f}")
    if cosine_sim is not None:
        parts.append(f"Cosine {cosine_sim:.2f}")
    if tanimoto is not None:
        parts.append(f"Tanimoto {tanimoto:.2f}")
    header_text = "  |  ".join(parts)

    if header_text:
        # Parse width/height for positioning
        m_w = re.search(r'width="(\d+)', svg)
        m_h = re.search(r'height="(\d+)', svg)
        W = int(m_w.group(1)) if m_w else 600
        H = int(m_h.group(1)) if m_h else 300

        # Anchor positions
        padding = 8
        text_y = 22  # from top
        if header_loc == "top-left":
            text_x = 10
        else:  # top-right
            # rough width estimate for background box
            est_text_px = 8 * len(header_text)  # ~8px per char
            text_x = max(W - est_text_px - 20, 10)

        # Background box (slight translucency), then text
        bg_w = max(8 * len(header_text) + 2 * padding, 120)
        bg_h = 24 + padding
        bg_x = max(text_x - padding, 0)
        bg_y = max(text_y - 18, 0)

        overlay = (
            f'<g id="metrics_overlay">'
            f'<rect x="{bg_x}" y="{bg_y}" width="{bg_w}" height="{bg_h}" '
            f'fill="white" opacity="0.7" stroke="none"/>'
            f'<text x="{text_x}" y="{text_y}" font-size="14" '
            f'font-family="DejaVu Sans, Arial, Helvetica" fill="black">{header_text}</text>'
            f"</g>"
        )

        # Insert overlay just before closing </svg> so it’s drawn on top
        insert_at = svg.rfind("</svg>")
        if insert_at == -1:
            insert_at = len(svg)
        svg = svg[:insert_at] + overlay + svg[insert_at:]

    if save_pth:
        p = Path(save_pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(svg, encoding="utf-8")

    display(SVG(svg))


def get_nature_hex_colors(extended: bool = True) -> list[str]:
    """
    Returns the 'nature' palette of hex colors.

    Args:
        extended: If True, include additional extended colors.
    """
    palette = ["#2664BF", "#34A89A", "#F69CA9", "#FBD399", "#AD95D1", "#FEA992"]
    if extended:
        palette += ["#AB8D8B", "#A8A9AB", "#6A4C93", "#C7EFCF", "#00CED1", "#FF6F61"]
    return palette


def color_generator(n_colors: int, cmap: str = "plotly"):  # noqa: C901
    """Generate a sequence of RGBA color tuples from a named colormap."""
    if cmap == "plotly":
        base = [
            (0.388, 0.431, 0.980, 1.0),
            (0.0, 0.8, 0.588, 1.0),
            (0.839, 0.153, 0.157, 1.0),
            (0.09, 0.745, 0.812, 1.0),
            (0.737, 0.741, 0.133, 1.0),
            (0.58, 0.404, 0.741, 1.0),
            (0.549, 0.337, 0.294, 1.0),
            (0.89, 0.467, 0.761, 1.0),
            (0.498, 0.498, 0.498, 1.0),
            (0.937, 0.333, 0.231, 1.0),
            (0.086, 0.465, 0.367, 1.0),
            (0.84, 0.8, 0.33, 1.0),
        ]
        return iter((base * ((n_colors // len(base)) + 1))[:n_colors])
    else:
        cmap_obj = plt.get_cmap(cmap)
        return (cmap_obj(i / (n_colors - 1)) for i in range(n_colors))


def rgb_to_hex(r, g, b) -> str:
    """Convert RGB or float [0,1] to hex color string."""
    if isinstance(r, float):
        r, g, b = [int(256 * v) for v in (r, g, b)]
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def get_palette(
    cmap: str = "nature", reversed_order: bool = False, as_hex: bool = False
) -> list:
    """Return a palette of colors by name."""
    if cmap == "nature":
        pal = get_nature_hex_colors()
    else:
        pal = list(color_generator(12, cmap=cmap))
        if as_hex:
            pal = [rgb_to_hex(*c[:3]) for c in pal]
    if reversed_order:
        pal = list(reversed(pal))
    return pal


def init_plotting(
    figsize: tuple[float, float] = (6, 4),
    font_scale: float = 1.0,
    style: str = "whitegrid",
    cmap: str = "nature",
    legend_outside: bool = False,
) -> None:
    """
    Initialize consistent plotting style across matplotlib and seaborn.

    Args:
        figsize: Default figure size (width, height) in inches.
        font_scale: Seaborn font scaling factor.
        style: Seaborn style (e.g., 'whitegrid', 'darkgrid').
        cmap: Name of palette to use ('nature', 'plotly', etc.).
        legend_outside: If True, wraps plt.legend to place legends outside plot area by default.
    """
    # Set base style and context
    plt.style.use("default")
    sns.set_style(style)
    sns.set_context("notebook", font_scale=font_scale)
    plt.rcParams["figure.figsize"] = figsize

    # Apply custom color palette
    pal = get_palette(cmap)
    sns.set_palette(pal)
    mpl.rcParams["axes.prop_cycle"] = cycler("color", pal)

    # If requested, wrap plt.legend to place legends outside by default
    if legend_outside:
        original_legend = plt.legend

        def legend_outside_fn(*args, **kwargs):
            kwargs.setdefault("loc", "upper left")
            kwargs.setdefault("bbox_to_anchor", (1.02, 1))
            return original_legend(*args, **kwargs)

        plt.legend = legend_outside_fn


def plot_similarity_kde(sim_mat: np.ndarray, figsize=(6, 4), palette_idx=(1,)):
    """
    Plot a KDE of the upper-triangle (unique) pairwise similarities.

    Args:
        sim_mat: 2D square array of pairwise similarities.
        figsize: Figure size in inches.
        palette_idx: Index or tuple of indices into nature palette for line color.
    """
    # Extract unique off-diagonals
    tri_i, tri_j = np.triu_indices_from(sim_mat, k=1)
    sims = sim_mat[tri_i, tri_j]
    # KDE
    kde = gaussian_kde(sims)
    xs = np.linspace(0, 1, 300)
    ys = kde(xs)
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    pal = get_nature_hex_colors()
    line_color = pal[palette_idx[0]]
    ax.plot(xs, ys, lw=2, color=line_color, label="KDE")
    ax.fill_between(xs, ys, color=line_color, alpha=0.3)
    # Bounds
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.axvline(1, color="k", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Estimated density")
    ax.set_title("KDE of unique pairwise similarities")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_dissimilarity_hist_kde(
    sim_mat: np.ndarray, figsize=(6, 4), bins=20, hist_idx=2, kde_idx=0
):
    """
    Plot histogram + KDE of dissimilarities (1 - similarity) from upper-triangle only.

    Args:
        sim_mat: 2D square array of pairwise similarities.
        figsize: Figure size in inches.
        bins: Number of histogram bins.
        hist_idx: Index into nature palette for histogram bars.
        kde_idx: Index into nature palette for KDE line.
    """
    # Extract and transform
    tri_i, tri_j = np.triu_indices_from(sim_mat, k=1)
    sims = sim_mat[tri_i, tri_j]
    dists = 1.0 - sims
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    pal = get_nature_hex_colors()
    # Histogram
    sns.histplot(
        dists,
        bins=bins,
        stat="density",
        edgecolor="w",
        ax=ax,
        color=pal[hist_idx],
        alpha=1.0,
    )
    # KDE
    sns.kdeplot(dists, linewidth=2, ax=ax, label="KDE", color=pal[kde_idx])
    ax.set_xlabel("1 − cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of unique dissimilarities")
    ax.legend()
    plt.tight_layout()
    plt.show()


def set_project_root(project_root: Path | str) -> None:
    """
    Point DreaMS plotting utils to your repo root so figures land under
    <PROJECT_ROOT>/misc/figures/..., not site-packages.
    """
    dplots.PROJECT_ROOT = Path(project_root)


def _figure_path(rel: str) -> Path:
    """
    Return absolute path under <PROJECT_ROOT>/misc/figures/<rel> and ensure parent exists.

    Example:
        _figure_path('figs/Fig2A_spectrum.svg')
        -> <PROJECT_ROOT>/misc/figures/figs/Fig2A_spectrum.svg
    """
    base = Path(getattr(dplots, "PROJECT_ROOT", Path.cwd()))
    p = base / "misc" / "figures" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def export_spectrum(msdata, i: int, rel_pth: str, **plot_kwargs) -> str:
    """
    Save a single spectrum using the SAME su.plot_spectrum function.

    Args:
        msdata: MSData-like object with .get_spectra(i) and .get_prec_mzs(i)
        i: index within msdata
        rel_pth: path relative to <PROJECT_ROOT>/misc/figures (e.g., 'figs/Fig2A_spectrum.svg')
        **plot_kwargs: forwarded to su.plot_spectrum (e.g., figsize, colors, xlim, etc.)

    Returns:
        Absolute path to the saved file as str.
    """
    abs_out = _figure_path(rel_pth)
    spec = msdata.get_spectra(i)
    prec = msdata.get_prec_mzs(i)
    su.plot_spectrum(
        spec,
        prec_mz=prec,
        save_pth=str(abs_out),  # pass ABSOLUTE so su.save_fig writes exactly here
        **plot_kwargs,
    )
    return str(abs_out)


def export_molecule(
    msdata,
    i: int,
    rel_pth: str,
    smiles_col: str = "smiles",
    width: int = 800,
    height: int = 600,
) -> str:
    """
    Save an RDKit depiction of the molecule at index i.
    Use '.svg' for vector suitable for Illustrator.

    Args:
        msdata: MSData-like object with .get_values(col, i)
        i: index within msdata
        rel_pth: path relative to <PROJECT_ROOT>/misc/figures (e.g., 'figs/Fig2A_molecule.svg')
        smiles_col: exact column name holding SMILES (e.g., 'smiles')
        width, height: canvas size in pixels

    Returns:
        Absolute path to the saved file as str.
    """
    out = _figure_path(rel_pth)

    smi = msdata.get_values(smiles_col, i)
    if isinstance(smi, (bytes, bytearray)):
        smi = smi.decode("utf-8", errors="ignore")
    if not smi:
        raise ValueError(f'No SMILES in column "{smiles_col}" at index {i}.')

    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        raise ValueError(f'RDKit could not parse SMILES "{smi}" at index {i}.')
    AllChem.Compute2DCoords(mol)

    if out.suffix.lower() == ".svg":
        d2d = rdMolDraw2D.MolDraw2DSVG(width, height)
        d2d.drawOptions().clearBackground = False  # cleaner for Illustrator
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        out.write_text(d2d.GetDrawingText(), encoding="utf-8")
    else:
        from rdkit.Chem.Draw import MolToImage

        img = MolToImage(mol, size=(width, height))
        img.save(str(out), dpi=(600, 600))
    return str(out)


def export_spectra_mol(
    msdata,
    i: int,
    basename: str = "Fig2A",
    spec_rel_dir: str = "figs",
    mol_rel_dir: str = "figs",
    spec_ext: str = "svg",
    mol_ext: str = "svg",
    smiles_col: str = "smiles",
    **plot_kwargs,
) -> Tuple[str, str]:
    """
    Convenience: save spectrum + molecule with matching names.

    Returns:
        (abs_spectrum_path, abs_molecule_path)
    """
    spec_rel = f"{spec_rel_dir}/{basename}_spectrum.{spec_ext}"
    mol_rel = f"{mol_rel_dir}/{basename}_molecule.{mol_ext}"
    spec_path = export_spectrum(msdata, i, spec_rel, **plot_kwargs)
    mol_path = export_molecule(msdata, i, mol_rel, smiles_col=smiles_col)
    return spec_path, mol_path
