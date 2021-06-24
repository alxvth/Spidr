#pragma once

#include <SpidrAnalysis.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <memory>
#include <optional>

namespace py = pybind11;

// This class introduces a unified&simplified constructor for SpidrAnalysis
class SpidrWrapper {
public:
	SpidrWrapper(
		distance_metric distMetric = distance_metric::METRIC_CHA,
		loc_Neigh_Weighting kernelType = loc_Neigh_Weighting::WEIGHT_UNIF,
		size_t numLocNeighbors = 0,
		size_t numHistBins = 0,
		knn_library aknnAlgType = knn_library::KNN_HNSW,
		int numIterations = 1000,
		int perplexity = 30,
		int exaggeration = 250,
		int expDecay = 70,
		bool forceCalcBackgroundFeatures = false);

	// compute knn dists and ids (and as part of that also the features)
	std::tuple<std::vector<int>, std::vector<float>> fit(
		py::array_t<float, py::array::c_style | py::array::forcecast> X,
		py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
		int imgWidth, int imgHight,
		std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal);

	py::array_t<float, py::array::c_style> transform();

	py::array_t<float, py::array::c_style> fit_transform(
		py::array_t<float, py::array::c_style | py::array::forcecast> X,
		py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
		int imgWidth, int imgHight,
		std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal);

	int get_nn() { return static_cast<int>(_nn); }
	int get_perplexity() { return _perplexity; }
	int get_numIterations() { return _numIterations; }

private:
	// utility function to circumvent code duplication in fit() and fit_transform()
	void compute_fit(
		py::array_t<float, py::array::c_style | py::array::forcecast> X,
		py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
		int imgWidth, int imgHight,
		std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal);


	std::unique_ptr<SpidrAnalysis> _SpidrAnalysis;

	feature_type _featType;
	loc_Neigh_Weighting _kernelType;
	size_t _numLocNeighbors;
	size_t _numHistBins;
	knn_library _aknnAlgType;
	distance_metric _distMetric;
	int _numIterations;
	int _perplexity;
	int _exaggeration;
	int _expDecay;
	bool _forceCalcBackgroundFeatures;

	size_t _numDims;
	py::ssize_t _numPoints;
	ImgSize _imgSize;

	bool _fitted;
	size_t _nn;
};

