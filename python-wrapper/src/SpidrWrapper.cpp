#include <pybind11/pybind11.h>
#include <SpidrAnalysis.h>

namespace py = pybind11;

PYBIND11_MODULE(SpidrWrapper, m) {
	m.doc() = "Spidr";

	py::class_<SpidrAnalysis> spidrAnalysis(m, "SpidrAnalysis");

	// Add the options from setupData and initializeAnalysisSettings
	// Aim for the same interface as sklear, like nptsne or umap-learn also did
	// Also add some python code like in nptsne/src/nptsne/hsne_analysis/analysis_model.py 
	//  in order to call umap
	spidrAnalysis.def(py::init<>());
}