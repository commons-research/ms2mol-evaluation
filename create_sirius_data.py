import os
from pathlib import Path

from matchms.exporting import save_as_mgf

from ms2mol_evaluation.isdb import filter_massspecgym_spectra, load_isdb
from ms2mol_evaluation.massspecgym import load_massspecgym, to_spectra


def main():
    massspecgym = load_massspecgym()
    spectra = to_spectra(massspecgym)
    isdb = load_isdb()
    spectra = filter_massspecgym_spectra(spectra, isdb, hydrogen_adduct_only=False)

    # spectra = spectra[:100]  # for testing purposes
    spectra_orbitrap = [s for s in spectra if s.get("instrument_type") == "Orbitrap"]
    spectra_qtof = [s for s in spectra if s.get("instrument_type") == "QTOF"]

    for s in spectra_orbitrap:
        s.set("ms_level", 2)
        s.set("formula", None)
        s.set("precursor_formula", None)
        s.set("feature_id", s.get("identifier"))

    for s in spectra_qtof:
        s.set("ms_level", 2)
        s.set("formula", None)
        s.set("precursor_formula", None)
        s.set("feature_id", s.get("identifier"))

    output_path = Path("data/sirius")
    os.makedirs(output_path, exist_ok=True)
    output_file_orbi = output_path / Path("sirius_orbitrap.mgf")
    output_file_qtof = output_path / Path("sirius_qtof.mgf")
    save_as_mgf(spectra_orbitrap, str(output_file_orbi), file_mode="w")
    save_as_mgf(spectra_qtof, str(output_file_qtof), file_mode="w")


if __name__ == "__main__":
    main()
