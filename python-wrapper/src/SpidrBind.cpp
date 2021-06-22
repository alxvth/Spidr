#include <pybind11/pybind11.h>
#include "SpidrWrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(SpidrWrapper, m) {
	m.doc() = "SpidrWrapper";

	// ENUMS
	py::enum_<distance_metric>(m, "KnnAlgorithm", "Distance metric, the choice of distance will set the feature type: scalar, hsitogram or point cloud")
		.value("dist.QF_hist", distance_metric::METRIC_QF)
		.value("dist.EMD_hist", distance_metric::METRIC_EMD)
		.value("dist.Hel_hist", distance_metric::METRIC_HEL)
		.value("dist.Chamfer_pc", distance_metric::METRIC_CHA)
		.value("dist.Hausdorff_pc", distance_metric::METRIC_HAU)
		.value("dist.Euclidean_scal", distance_metric::METRIC_EUC);

	py::enum_<loc_Neigh_Weighting>(m, "KnnAlgorithm", "Distance metric, the choice of distance will set the feature type: scalar, hsitogram or point cloud")
		.value("weight.uniform", loc_Neigh_Weighting::WEIGHT_UNIF)
		.value("weight.bino", loc_Neigh_Weighting::WEIGHT_BINO)
		.value("weight.gauss", loc_Neigh_Weighting::WEIGHT_GAUS);

	py::enum_<knn_library>(m, "KnnAlgorithm", "Distance metric, the choice of distance will set the feature type: scalar, hsitogram or point cloud")
		.value("knn.hnsw", knn_library::KNN_HNSW)
		.value("knn.exact", knn_library::EXACT);

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
		py::arg("imgHight"),
		py::arg("backgroundIDsGlobal") = py::none());

	spidrAnalysis.def("transform", &SpidrWrapper::transform, "Compute embedding, fit() must have been called previously");

	spidrAnalysis.def("fit_transform", &SpidrWrapper::fit_transform, "Compute embedding, calls fit()",
		py::arg("X"),
		py::arg("pointIDsGlobal"),
		py::arg("imgWidth"),
		py::arg("imgHight"),
		py::arg("backgroundIDsGlobal") = py::none());

}