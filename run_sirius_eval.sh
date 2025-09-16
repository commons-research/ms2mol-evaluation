#! /bin/sh

echo "Running SIRIUS evaluation...";
sirius --input data/sirius/sirius_orbitrap.mgf \
    --output data/sirius/sirius_orbitrap \
    config \
    --IsotopeSettings.filter=true \
    --CandidateFormulas=, \
    --FormulaSettings.enforced=H,C,N,O,P \
    --Timeout.secondsPerInstance=0 \
    --AlgorithmProfile=orbitrap \
    --SpectralMatchingMassDeviation.allowedPeakDeviation=10.0ppm \
    --AdductSettings.ignoreDetectedAdducts=false \
    --AdductSettings.prioritizeInputFileAdducts=true \
    --UseHeuristic.useHeuristicAboveMz=300 \
    --IsotopeMs2Settings=IGNORE \
    --MS2MassDeviation.allowedMassDeviation=10.0ppm \
    --SpectralMatchingMassDeviation.allowedPrecursorDeviation=10.0ppm \
    --FormulaSearchSettings.performDeNovoBelowMz=400.0 \
    --FormulaSearchSettings.applyFormulaConstraintsToDatabaseCandidates=false \
    --EnforceElGordoFormula=true \
    --NumberOfCandidatesPerIonization=1 \
    --AdductSettings.fallback=[[M+H]+,[M+Na]+] \
    --FormulaSearchSettings.performBottomUpAboveMz=0 \
    --FormulaSearchSettings.applyFormulaConstraintsToBottomUp=false \
    --UseHeuristic.useOnlyHeuristicAboveMz=650 \
    --FormulaSearchDB=, \
    --Timeout.secondsPerTree=0 \
    --AdductSettings.enforced=, \
    --FormulaSettings.detectable=B,S,Cl,Se,Br \
    --NumberOfCandidates=10 \
    --FormulaResultThreshold=true \
    --ExpansiveSearchConfidenceMode.confidenceScoreSimilarityMode=OFF \
    --StructureSearchDB=LOTUS \
    --RecomputeResults=false \
    spectra-search formulas fingerprints classes structures summaries --chemvista --feature-quality-summary --full-summary ;


sirius \
    --input data/sirius/sirius_qtof.mgf \
    --output data/sirius/sirius_qtof \
    config \
    --IsotopeSettings.filter=true \
    --CandidateFormulas=, \
    --FormulaSettings.enforced=H,C,N,O,P \
    --Timeout.secondsPerInstance=0 \
    --AlgorithmProfile=orbitrap \
    --SpectralMatchingMassDeviation.allowedPeakDeviation=10.0ppm \
    --AdductSettings.ignoreDetectedAdducts=false \
    --AdductSettings.prioritizeInputFileAdducts=true \
    --UseHeuristic.useHeuristicAboveMz=300 \
    --IsotopeMs2Settings=IGNORE \
    --MS2MassDeviation.allowedMassDeviation=10.0ppm \
    --SpectralMatchingMassDeviation.allowedPrecursorDeviation=10.0ppm \
    --FormulaSearchSettings.performDeNovoBelowMz=400.0 \
    --FormulaSearchSettings.applyFormulaConstraintsToDatabaseCandidates=false \
    --EnforceElGordoFormula=true \
    --NumberOfCandidatesPerIonization=1 \
    --AdductSettings.fallback=[[M+H]+,[M+Na]+] \
    --FormulaSearchSettings.performBottomUpAboveMz=0 \
    --FormulaSearchSettings.applyFormulaConstraintsToBottomUp=false \
    --UseHeuristic.useOnlyHeuristicAboveMz=650 \
    --FormulaSearchDB=, \
    --Timeout.secondsPerTree=0 \
    --AdductSettings.enforced=, \
    --FormulaSettings.detectable=B,S,Cl,Se,Br \
    --NumberOfCandidates=10 \
    --FormulaResultThreshold=true \
    --ExpansiveSearchConfidenceMode.confidenceScoreSimilarityMode=OFF \
    --StructureSearchDB=LOTUS \
    --RecomputeResults=false \
    spectra-search formulas fingerprints classes structures summaries --chemvista --feature-quality-summary --full-summary ; 