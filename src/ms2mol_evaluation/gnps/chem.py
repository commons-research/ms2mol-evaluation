from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize as std

_norm = std.Normalizer()
_reion = std.Reionizer()
_unch = std.Uncharger()
_lfc = std.LargestFragmentChooser()
_te = std.TautomerEnumerator()
_md = std.MetalDisconnector()


def standardize_mol(mol: Chem.Mol | None) -> Chem.Mol | None:
    """Largest fragment, disconnect metals, normalize, reionize, uncharge, canonical tautomer, strip isotopes."""
    if mol is None:
        return None
    try:
        mol = _lfc.choose(mol)
        mol = _md.Disconnect(mol)
        mol = _norm.normalize(mol)
        mol = _reion.reionize(mol)
        mol = _unch.uncharge(mol)
        mol = _te.Canonicalize(mol)
        for a in mol.GetAtoms():
            if a.GetIsotope():
                a.SetIsotope(0)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def mol_from_smiles_clean(smiles: str | None) -> Chem.Mol | None:
    """SMILES → RDKit Mol → standardized Mol; returns None if unparsable."""
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    return standardize_mol(mol)


def classical_tanimoto(smiles1: str | None, smiles2: str | None) -> float | None:
    """Morgan r=2, 2048 bits Tanimoto on standardized molecules. Returns None if either invalid."""
    m1 = mol_from_smiles_clean(smiles1)
    m2 = mol_from_smiles_clean(smiles2)
    if not m1 or not m2:
        return None
    f1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    f2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(f1, f2))
