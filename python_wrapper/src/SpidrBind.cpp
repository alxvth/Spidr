#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SpidrWrapper.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(SpidrWrapper, m) {
	m.doc() = "SpidrWrapper";

	// ENUMS
	py::enum_<distance_metric>(m, "DistMetric", "Distance metric, the choice of distance will set the feature type: scalar, histogram or point cloud")
		.value("QF_hist", distance_metric::METRIC_QF)
		.value("Hel_hist", distance_metric::METRIC_HEL)
		.value("Chamfer_pc", distance_metric::METRIC_CHA)
		.value("Hausdorff_pc", distance_metric::METRIC_HAU)
		.value("Morans_I", distance_metric::METRIC_EUC)
		.value("Bhattacharyya", distance_metric::METRIC_BHATTACHARYYA);

	py::enum_<loc_Neigh_Weighting>(m, "WeightLoc", "Distance metric, the choice of distance will set the feature type: scalar, histogram or point cloud")
		.value("uniform", loc_Neigh_Weighting::WEIGHT_UNIF)
		.value("bino", loc_Neigh_Weighting::WEIGHT_BINO)
		.value("gauss", loc_Neigh_Weighting::WEIGHT_GAUS);

	py::enum_<knn_library>(m, "KnnAlgorithm", "Distance metric, the choice of distance will set the feature type: scalar, histogram or point cloud")
		.value("hnsw", knn_library::KNN_HNSW)
		.value("exact_knn", knn_library::KKN_EXACT)
		.value("full_dist_matrix", knn_library::FULL_DIST_BRUTE_FORCE);

	// MAIN WRAPPER: here SpidrWrapper, on python side SpidrAnalysis
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


	spidrAnalysis.def("fit", &SpidrWrapper::fit, "Compute kNN dists and indices and return them",
		py::arg("X"),
		py::arg("pointIDsGlobal"),
		py::arg("imgWidth"),
		py::arg("imgHeight"),
		py::arg("backgroundIDsGlobal") = py::none());

	spidrAnalysis.def("compute_fit", &SpidrWrapper::compute_fit, "Compute kNN dists and indices, do not return anything",
		py::arg("X"),
		py::arg("pointIDsGlobal"),
		py::arg("imgWidth"),
		py::arg("imgHeight"),
		py::arg("backgroundIDsGlobal") = py::none());

	spidrAnalysis.def("transform", &SpidrWrapper::transform, "Compute embedding and return it, fit() must have been called previously");

	spidrAnalysis.def("compute_transform", &SpidrWrapper::compute_transform, "Compute embedding but do not return anything, fit() must have been called previously");

	spidrAnalysis.def("fit_transform", &SpidrWrapper::fit_transform, "Compute embedding and return it",
		py::arg("X"),
		py::arg("pointIDsGlobal"),
		py::arg("imgWidth"),
		py::arg("imgHeight"),
		py::arg("backgroundIDsGlobal") = py::none());

	spidrAnalysis.def("set_kNN", &SpidrWrapper::set_kNN, "Compute embedding, calls fit()",
		py::arg("knn_indices"),
		py::arg("knn_distances"));

	spidrAnalysis.def("get_kNN", &SpidrWrapper::get_kNN, "Returns the kNN dists and indices");

	spidrAnalysis.def_property_readonly("perplexity", &SpidrWrapper::get_perplexity, "t-SNE perplexity");
	spidrAnalysis.def_property_readonly("iterations", &SpidrWrapper::get_numIterations, "t-SNE iterations");
	spidrAnalysis.def_property_readonly("nn", &SpidrWrapper::get_nn, "Number of nearest neighbors");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif


}
