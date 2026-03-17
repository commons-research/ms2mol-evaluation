SIRIUS_ORBITRAP_COMMAND = "config \
    --AlgorithmProfile=orbitrap \
    --IdentitySearchSettings.precursorDeviation=20.0ppm \
    --MS2MassDeviation.allowedMassDeviation=0.01Da \
    --SpectralSearchDB=METACYC,BloodExposome,CHEBI,COCONUT,FooDB,GNPS,HMDB,HSDB,KEGG,KNAPSACK,LOTUS,LIPIDMAPS,MACONDA,MESH,MiMeDB,NORMAN,PLANTCYC,PUBCHEMANNOTATIONBIO,PUBCHEMANNOTATIONDRUG,PUBCHEMANNOTATIONFOOD,PUBCHEMANNOTATIONSAFETYANDTOXIC,SUPERNATURAL,TeroMol,YMDB \
    --AdductSettings.fallback=[[M+H]+,[M+Na]+,[M+K]+] \
    --FormulaSettings.enforced=H,C,N,O,P \
    --FormulaSearchSettings.performBottomUpAboveMz=0 \
    --FormulaSearchDB=, \
    --StructureSearchDB=METACYC,BloodExposome,CHEBI,COCONUT,FooDB,GNPS,HMDB,HSDB,KEGG,KNAPSACK,LOTUS,LIPIDMAPS,MACONDA,MESH,MiMeDB,NORMAN,PLANTCYC,PUBCHEMANNOTATIONBIO,PUBCHEMANNOTATIONDRUG,PUBCHEMANNOTATIONFOOD,PUBCHEMANNOTATIONSAFETYANDTOXIC,SUPERNATURAL,TeroMol,YMDB \
    spectra-search formulas fingerprints classes structures summaries --chemvista --feature-quality-summary --full-summary"


SIRIUS_QTOF_COMMAND = "config \
    --AlgorithmProfile=qtof \
    --IdentitySearchSettings.precursorDeviation=20.0ppm \
    --MS2MassDeviation.allowedMassDeviation=0.01Da \
    --SpectralSearchDB=METACYC,BloodExposome,CHEBI,COCONUT,FooDB,GNPS,HMDB,HSDB,KEGG,KNAPSACK,LOTUS,LIPIDMAPS,MACONDA,MESH,MiMeDB,NORMAN,PLANTCYC,PUBCHEMANNOTATIONBIO,PUBCHEMANNOTATIONDRUG,PUBCHEMANNOTATIONFOOD,PUBCHEMANNOTATIONSAFETYANDTOXIC,SUPERNATURAL,TeroMol,YMDB \
    --AdductSettings.fallback=[[M+H]+,[M+Na]+,[M+K]+] \
    --FormulaSettings.enforced=H,C,N,O,P \
    --FormulaSearchSettings.performBottomUpAboveMz=0 \
    --FormulaSearchDB=, \
    --StructureSearchDB=METACYC,BloodExposome,CHEBI,COCONUT,FooDB,GNPS,HMDB,HSDB,KEGG,KNAPSACK,LOTUS,LIPIDMAPS,MACONDA,MESH,MiMeDB,NORMAN,PLANTCYC,PUBCHEMANNOTATIONBIO,PUBCHEMANNOTATIONDRUG,PUBCHEMANNOTATIONFOOD,PUBCHEMANNOTATIONSAFETYANDTOXIC,SUPERNATURAL,TeroMol,YMDB \
    spectra-search formulas fingerprints classes structures summaries --chemvista --feature-quality-summary --full-summary"
