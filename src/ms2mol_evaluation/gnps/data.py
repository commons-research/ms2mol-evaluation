import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import dreams.utils.spectra as su
import networkx as nx
import numpy as np
import pandas as pd
from dreams.definitions import PRECURSOR_MZ, SPECTRUM
from dreams.utils.spectra import unpad_peak_list
from matchms.exporting import save_as_mgf
from matchms.importing import load_from_mgf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph

from ms2mol_evaluation.gnps.chem import classical_tanimoto


def _assign_scalar_attr(obj: dict, key: str, value) -> None:
    """
    Assign only simple scalars supported by GraphML. Skip None, NaN/Inf, lists/arrays, and empty/'nan'/'none' strings.
    Convert numpy scalars to Python scalars.
    """
    if value is None:
        return
    # skip containers & arrays
    if isinstance(value, (list, tuple, dict, set, np.ndarray)):
        return
    # convert numpy 0-d scalars
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    # drop non-finite floats
    if isinstance(value, float) and not np.isfinite(value):
        return
    # drop empty-ish strings
    if isinstance(value, str) and value.strip().lower() in ("", "nan", "none"):
        return
    # keep: str, bool, int, float
    if isinstance(value, (str, bool, int, float)):
        obj[key] = value


def build_query_knn_graph(embs_q: np.ndarray, k: int, thld: float) -> nx.Graph:
    """k-NN on query embeddings → cosine distance -> similarity (1 - d) → threshold → undirected graph."""
    A = kneighbors_graph(
        embs_q, k, mode="distance", metric="cosine", include_self=False
    )
    A = A.toarray().astype(float)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0.0:
                A[i, j] = 1.0 - A[i, j]
            if A[i, j] < thld:
                A[i, j] = 0.0
    G = nx.from_numpy_array(A)
    for i in G.nodes():
        G.nodes[i]["node_type"] = "query"
        G.nodes[i]["id"] = f"Q_{i}"
    return G


def annotate_edges_modcos_qq(G: nx.Graph, msdata_q, mz_tolerance: float = 0.05) -> None:
    """
    Add node/edge info for Cytoscape:
      - Nodes (queries): copy all scalar-safe msdata_q fields into node attrs; set a fallback 'label'.
      - Edges (Q–Q): 'DreaMS_similarity', set 'edge_type'='Q-Q', compute and store 'modified_cosine_similarity'.
    """
    # --- 1) node attributes (query nodes) ---
    for u, d in G.nodes(data=True):
        # only attach for query nodes that map to msdata_q row indices
        if d.get("node_type", "query") != "query":
            continue
        if not isinstance(u, int) or u < 0 or u >= len(msdata_q):
            continue

        # pull row without plotting payloads
        try:
            row = msdata_q.at(u, plot_spec=False, plot_mol=False)
        except Exception:
            row = {}

        # attach scalar-safe attrs from whatever columns exist
        for k, v in row.items():
            _assign_scalar_attr(d, k, v)

        # ensure a readable label (generic, column-agnostic)
        if "label" not in d or not isinstance(d["label"], str) or not d["label"]:
            _assign_scalar_attr(d, "label", d.get("id", f"Q_{u}"))

    # --- 2) edge annotations (Q–Q) ---
    cos_sim_pl = su.PeakListModifiedCosine(mz_tolerance=mz_tolerance)
    for u, v, d in G.edges(data=True):
        if "weight" in d:
            d["DreaMS_similarity"] = float(d["weight"])
            del d["weight"]
        d["edge_type"] = "Q-Q"
        try:
            mc = cos_sim_pl(
                spec1=msdata_q[SPECTRUM][u],
                prec_mz1=msdata_q[PRECURSOR_MZ][u],
                spec2=msdata_q[SPECTRUM][v],
                prec_mz2=msdata_q[PRECURSOR_MZ][v],
            )
            if isinstance(mc, (int, float)) and np.isfinite(float(mc)):
                d["modified_cosine_similarity"] = float(mc)
        except Exception:
            pass


def find_q2lib_neighbors(
    embs_q: np.ndarray,
    embs_lib: np.ndarray,
    sim_thld: float,
    topk_per_q: Optional[int] = 5,
) -> tuple[Dict[int, List[tuple[int, float]]], set[int]]:
    """For each query qi, return [(lib_idx, sim)] with sim ≥ sim_thld, sorted desc and truncated to topk_per_q."""
    sims = cosine_similarity(embs_q, embs_lib)  # (nQ, nL)
    q2lib: Dict[int, List[tuple[int, float]]] = {}
    lib_keep: set[int] = set()
    for qi in range(sims.shape[0]):
        hits = np.where(sims[qi] >= sim_thld)[0]
        if hits.size:
            hits = hits[np.argsort(sims[qi, hits])[::-1]]
            if topk_per_q is not None:
                hits = hits[:topk_per_q]
            q2lib[qi] = [(int(j), float(sims[qi, j])) for j in hits]
            lib_keep.update(hits.tolist())
    return q2lib, lib_keep


def augment_with_library_nodes(
    G: nx.Graph,
    q2lib: Dict[int, List[tuple[int, float]]],
    lib_keep: set[int],
    msdata_q,
    msdata_lib,
    safe_lib_cols: Optional[Set[str]] = None,  # kept for signature; None = no filtering
    mz_tolerance: float = 0.05,
) -> None:
    """Add only selected library nodes and Q-L edges with similarities."""
    offset = len(msdata_q)
    lib_idx_to_node: Dict[int, int] = {}

    # add library nodes
    for j in sorted(lib_keep):
        nid = offset + j
        G.add_node(nid)
        G.nodes[nid]["node_type"] = "library"
        G.nodes[nid]["id"] = f"L_{j}"
        row = msdata_lib.at(j, plot_spec=False, plot_mol=False)
        for k, v in row.items():
            if safe_lib_cols is not None and k not in safe_lib_cols:
                continue
            _assign_scalar_attr(G.nodes[nid], k, v)
        # label
        label = row.get("IDENTIFIER", None) or row.get("INCHIKEY", None) or f"L_{j}"
        _assign_scalar_attr(G.nodes[nid], "label", label)
        lib_idx_to_node[j] = nid

    # edges Q-L with sims + optional mod cosine
    cos_sim_pl = su.PeakListModifiedCosine(mz_tolerance=mz_tolerance)
    for qi, hits in q2lib.items():
        u = qi
        for lj, sim in hits:
            v = lib_idx_to_node[lj]
            if not G.has_edge(u, v):
                G.add_edge(u, v)
            d = G[u][v]
            d["edge_type"] = "Q-L"
            if d.get("DreaMS_similarity", 0.0) < sim:
                d["DreaMS_similarity"] = float(sim)
            try:
                mc = cos_sim_pl(
                    spec1=msdata_q[SPECTRUM][u],
                    prec_mz1=msdata_q[PRECURSOR_MZ][u],
                    spec2=msdata_lib[SPECTRUM][lj],
                    prec_mz2=msdata_lib[PRECURSOR_MZ][lj],
                )
                if isinstance(mc, (int, float)) and np.isfinite(float(mc)):
                    d["modified_cosine_similarity"] = float(mc)
            except Exception:
                pass


def tag_top1_assignment(
    G,
    embs_q,
    embs_lib,
    msdata_q,
    msdata_lib,
    sim_thld: float,
    overwrite_smiles: bool = False,
    mz_tolerance: float = 0.05,  # <-- NEW (default)
) -> None:
    """Tag each query node with top-1 library hit info (keeps assignment by DreaMS sim)."""
    sims = cosine_similarity(embs_q, embs_lib)
    top1_idx = sims.argmax(axis=1)
    top1_sim = sims[np.arange(sims.shape[0]), top1_idx]

    lib_cols = set(msdata_lib.columns())
    lib_has_ident = "IDENTIFIER" in lib_cols
    lib_has_ikey = "INCHIKEY" in lib_cols
    lib_has_smi = "smiles" in lib_cols

    cos_sim_pl = su.PeakListModifiedCosine(mz_tolerance=mz_tolerance)

    for qi in range(sims.shape[0]):
        sim = float(top1_sim[qi])
        lj = int(top1_idx[qi])

        if sim >= sim_thld:
            # identifier
            ident = None
            if lib_has_ident:
                try:
                    ident = msdata_lib.get_values("IDENTIFIER", lj)
                except Exception:
                    ident = None
            if not ident and lib_has_ikey:
                try:
                    ident = msdata_lib.get_values("INCHIKEY", lj)
                except Exception:
                    ident = None
            if not isinstance(ident, str) or ident.strip().lower() in (
                "",
                "nan",
                "none",
            ):
                ident = f"L_{lj}"

            G.nodes[qi]["has_lib_hit"] = "hit"
            G.nodes[qi]["DreaMS_best_lib_similarity"] = sim
            G.nodes[qi]["best_lib_identifier"] = ident

            try:
                mc = cos_sim_pl(
                    spec1=msdata_q[SPECTRUM][qi],
                    prec_mz1=msdata_q[PRECURSOR_MZ][qi],
                    spec2=msdata_lib[SPECTRUM][lj],
                    prec_mz2=msdata_lib[PRECURSOR_MZ][lj],
                )
                if isinstance(mc, (int, float)) and np.isfinite(float(mc)):
                    G.nodes[qi]["modified_cosine_best_lib"] = float(mc)
            except Exception:
                pass

            if lib_has_smi:
                try:
                    lib_smiles = msdata_lib.get_values("smiles", lj)
                except Exception:
                    lib_smiles = None
                if isinstance(lib_smiles, str) and lib_smiles.strip().lower() not in (
                    "",
                    "nan",
                    "none",
                ):
                    existing = G.nodes[qi].get("smiles", None)
                    exists = isinstance(
                        existing, str
                    ) and existing.strip().lower() not in ("", "nan", "none")
                    if overwrite_smiles or not exists:
                        G.nodes[qi]["smiles"] = lib_smiles
        else:
            G.nodes[qi]["has_lib_hit"] = "no_hit"


def export_graphs(G: nx.Graph, out_base):
    """Write GraphML directly. Attributes must already be scalar-safe."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    graphml_path = out_base.with_suffix(".graphml")
    nx.write_graphml(G, graphml_path)

    print(f"Saved GraphML → {graphml_path}")
    return graphml_path


def to_clean_str(x: Any) -> str | None:
    """Bytes→str, strip; None for common placeholders."""
    missing_strings = {"none"}
    if x is None:
        return None
    if isinstance(x, bytes):
        try:
            x = x.decode()
        except Exception:
            x = x.decode("latin1", errors="ignore")
    s = str(x).strip()
    if s.lower() in missing_strings:
        return None
    return s


def inchikey_first_block(x: Any) -> str | None:
    """Uppercase INCHIKEY and return the block before the first '-'. Require ≥14 chars."""
    s = to_clean_str(x)
    if s is None:
        return None
    blk = s.upper().split("-")[0]
    return blk if len(blk) >= 14 else None


def build_rank1_annotations(
    df_matches: pd.DataFrame, msdata_q, msdata_lib
) -> pd.DataFrame:
    """
    From df_matches (with columns: query_index, library_index, rank, DreaMS_similarity, Modified_cosine_similarity, library_IDENTIFIER),
    build a dataframe for rank==1 pairs with cleaned SMILES/INCHIKEYs, an annotation match flag, and classical tanimoto.
    """
    df1 = df_matches[df_matches["rank"] == 1].copy()

    # Pull raw annotations from MSData
    df1["q_SMILES_raw"] = df1["query_index"].apply(
        lambda i: msdata_q.get_values("smiles", int(i))
    )
    df1["l_SMILES_raw"] = df1["library_index"].apply(
        lambda j: msdata_lib.get_values("smiles", int(j))
    )
    df1["q_INCHIKEY_raw"] = df1["query_index"].apply(
        lambda i: msdata_q.get_values("INCHIKEY", int(i))
    )
    df1["l_INCHIKEY_raw"] = df1["library_index"].apply(
        lambda j: msdata_lib.get_values("INCHIKEY", int(j))
    )

    # Clean + normalize
    df1["q_SMILES"] = df1["q_SMILES_raw"].map(to_clean_str)
    df1["l_SMILES"] = df1["l_SMILES_raw"].map(to_clean_str)
    df1["q_IK_block"] = df1["q_INCHIKEY_raw"].map(inchikey_first_block)
    df1["l_IK_block"] = df1["l_INCHIKEY_raw"].map(inchikey_first_block)

    # Keep only pairs with both INCHIKEYs
    annot = df1.dropna(subset=["q_IK_block", "l_IK_block"]).copy()

    # Annotation agreement flag
    annot["annotation_match"] = annot["q_IK_block"] == annot["l_IK_block"]

    # Classical Tanimoto (Morgan r=2) — name it as requested
    annot["Tanimoto_similarity"] = annot.apply(
        lambda r: classical_tanimoto(r["q_SMILES"], r["l_SMILES"]), axis=1
    )

    return annot


def annotate_mgf_with_label(
    input_mgf: pathlib.Path, output_mgf: pathlib.Path, label_fn
):
    """
    Reads spectra from `input_mgf`, computes label_fn(metadata) for each,
    writes new MGF with a "LABEL=<float>" entry in each block.

    Uses `Spectrum.set()` to properly update metadata without altering the spectra.
    """
    # Load all spectra
    specs = list(load_from_mgf(str(input_mgf)))

    # Annotate each spectrum
    for spec in specs:
        lbl = float(label_fn(spec.metadata))
        spec.set("LABEL", str(lbl))

    # Save back out
    save_as_mgf(specs, str(output_mgf))


def find_lsh_diverse_index(
    msdata,
    hashes: np.ndarray,
    cluster_size: int = 5,
    delta_mz: float = 10.0,
    count_nan_as_category: bool = True,
    require_multi_instrument: bool = True,
    require_multi_energy: bool = True,
    return_details: bool = False,
) -> Any:
    """
    Return index `i` into `lsh_counts[lsh_counts==cluster_size].index` of the first cluster that:
    same INCHIKEY, ≥2 instruments, ≥2 energies, and ≥1 spectrum with max(m/z) ≥ precursor+delta_mz.
    """
    # candidate clusters (keep pandas ordering) + silence future warning by inferring index objects
    lsh_counts = pd.Series(hashes).value_counts()
    try:
        lsh_counts.index = lsh_counts.index.infer_objects()
    except Exception:
        pass
    target_hashes = lsh_counts[lsh_counts == cluster_size].index

    # preload columns
    inchis_all = np.asarray(msdata.get_values("INCHIKEY"), dtype=object)
    inst_all = np.asarray(msdata.get_values("INSTRUMENT_TYPE"), dtype=object)
    ce_all = np.asarray(msdata.get_values("COLLISION_ENERGY"), dtype=object)
    prec_all = np.asarray(msdata.get_values("precursor_mz"), dtype=float)

    def _norm_labels(arr: np.ndarray) -> pd.Series:
        out: List[object] = []
        for x in arr:
            if isinstance(x, bytes):
                x = x.decode("utf-8", errors="ignore")
            if x is None:
                out.append(np.nan)
            elif isinstance(x, str):
                t = x.strip()
                out.append(np.nan if (t == "" or t.lower() == "nan") else t)
            else:
                out.append(x)
        return pd.Series(out, dtype=object)

    def _norm_energy(s: pd.Series) -> pd.Series:
        return s.map(
            lambda x: (
                np.nan if (isinstance(x, str) and x.strip().lower() == "nan") else x
            )
        )

    def _has_heavy_peak(idxs: List[int], delta: float) -> bool:
        for j in idxs:
            j = int(j)
            prec = prec_all[j]
            if not np.isfinite(prec):
                continue
            pl = unpad_peak_list(msdata.get_spectra()[j])  # (2, n_peaks)
            if pl.shape[1] and float(pl[0].max()) >= float(prec) + float(delta):
                return True
        return False

    dropna_flag = not count_nan_as_category
    chosen_pos: Optional[int] = None

    for pos, h in enumerate(target_hashes):
        idx_tmp = np.where(hashes == h)[0]
        ilist = idx_tmp.tolist()

        if pd.Series(inchis_all[ilist], dtype=object).nunique(dropna=True) != 1:
            continue
        if (
            require_multi_instrument
            and _norm_labels(inst_all[ilist]).nunique(dropna=dropna_flag) < 2
        ):
            continue
        if (
            require_multi_energy
            and _norm_energy(pd.Series(ce_all[ilist], dtype=object)).nunique(
                dropna=dropna_flag
            )
            < 2
        ):
            continue
        if not _has_heavy_peak(ilist, delta_mz):
            continue

        chosen_pos = pos
        break

    if chosen_pos is None:
        raise ValueError(
            f"No cluster met criteria (size=={cluster_size}, same INCHIKEY, "
            f"multi-instrument, multi-energy, heavy peak ≥ precursor+{delta_mz})."
        )

    if not return_details:
        return chosen_pos

    chosen_hash = target_hashes[chosen_pos]
    idx = np.where(hashes == chosen_hash)[0]
    details: Dict[str, Any] = {
        "lsh": chosen_hash,
        "idx": idx,
        "instruments": list(_norm_labels(inst_all[idx.tolist()]).unique()),
        "energies": list(pd.Series(ce_all[idx.tolist()], dtype=object).unique()),
    }
    return chosen_pos, details
