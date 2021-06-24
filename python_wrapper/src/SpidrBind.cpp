#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SpidrWrapper.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(SpidrWrapper, m) {
	m.doc() = "SpidrWrapper";

	// ENUMS
	py::enum_<distance_metric>(m, "DistMetric", "Distance metric, the choice of distance will set the feature type: scalar, hsitogram or point cloud")
		.value("QF_hist", distance_metric::METRIC_QF)
		.value("EMD_hist", distance_metric::METRIC_EMD)
		.value("Hel_hist", distance_metric::METRIC_HEL)
		.value("Chamfer_pc", distance_metric::METRIC_CHA)
		.value("Hausdorff_pc", distance_metric::METRIC_HAU)
		.value("Euclidean_scal", distance_metric::METRIC_EUC);

	py::enum_<loc_Neigh_Weighting>(m, "WeightLoc", "Distance metric, the choice of distance will set the feature type: scalar, hsitogram or point cloud")
		.value("uniform", loc_Neigh_Weighting::WEIGHT_UNIF)
		.value("bino", loc_Neigh_Weighting::WEIGHT_BINO)
		.value("gauss", loc_Neigh_Weighting::WEIGHT_GAUS);

	py::enum_<knn_library>(m, "KnnAlgorithm", "Distance metric, the choice of distance will set the feature type: scalar, hsitogram or point cloud")
		.value("hnsw", knn_library::KNN_HNSW)
		.value("exact", knn_library::EXACT);

	py::class_<SpidrWrapper> spidrAnalysis(m, "SpidrAnalysis");

	spidrAnalysis.def(py::init<distance_metric, loc_Neigh_Weighting, size_t, size_t, knn_library, int, int, int, int, bool>(), "Init SpidrLib",
		py::arg("distMetric") = distance_metric::METRIC_CHA,
		py::arg("kernelType") = loc_Neigh_Weighting::WEIGHT_UNIF,
		py::arg("numLocNeighbors") = 0,
		py::arg("numHistBins") = 0,
		py::arg("aknnAlgType") = knn_library::KNN_HNSW,
		py::arg("numIterations") = 1000,
		py::arg("perplexity") = 30,
		py::arg("exaggeration") = 250,
		py::arg("expDecay") = 70,
		py::arg("forceCalcBackgroundFeatures") = false);


	spidrAnalysis.def("fit", &SpidrWrapper::fit, "Compute kNN dists and indices",
		py::arg("X"),
		py::arg("pointIDsGlobal"),
		py::arg("imgWidth"),
		py::arg("imgHeight"),
		py::arg("backgroundIDsGlobal") = py::none());

	spidrAnalysis.def("transform", &SpidrWrapper::transform, "Compute embedding, fit() must have been called previously");

	spidrAnalysis.def("fit_transform", &SpidrWrapper::fit_transform, "Compute embedding, calls fit()",
		py::arg("X"),
		py::arg("pointIDsGlobal"),
		py::arg("imgWidth"),
		py::arg("imgHeight"),
		py::arg("backgroundIDsGlobal") = py::none());

	spidrAnalysis.def_property_readonly("perplexity", &SpidrWrapper::get_perplexity, "t-SNE perplexity");
	spidrAnalysis.def_property_readonly("iterations", &SpidrWrapper::get_numIterations, "t-SNE iterations");
	spidrAnalysis.def_property_readonly("nn", &SpidrWrapper::get_nn, "Number of nearest neighbors");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif


}
