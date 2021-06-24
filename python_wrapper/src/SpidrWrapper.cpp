#include "SpidrWrapper.h"

SpidrWrapper::SpidrWrapper(distance_metric distMetric,
	loc_Neigh_Weighting kernelType,
	size_t numLocNeighbors,
	size_t numHistBins,
	knn_library aknnAlgType,
	int numIterations,
	int perplexity,
	int exaggeration,
	int expDecay,
	bool forceCalcBackgroundFeatures
) : _kernelType(kernelType), _numHistBins(numHistBins), _aknnAlgType(aknnAlgType), _distMetric(distMetric), _numIterations(numIterations),
    _perplexity(perplexity), _exaggeration(exaggeration), _expDecay(expDecay), _forceCalcBackgroundFeatures(forceCalcBackgroundFeatures), _fitted(false) 
{
	if (numLocNeighbors <= 0)
		throw std::runtime_error("SpidrWrapper: Spatial Neighbors must be larger 0");
	else
		_numLocNeighbors = numLocNeighbors;

	// set _featType depending on aknnMetric
	switch (distMetric) {
	case distance_metric::METRIC_QF:
	case distance_metric::METRIC_EMD:
	case distance_metric::METRIC_HEL: 
		_featType = feature_type::TEXTURE_HIST_1D; break;
	case distance_metric::METRIC_CHA:
	case distance_metric::METRIC_HAU:
		_featType = feature_type::PCLOUD; break;
	case distance_metric::METRIC_EUC:
		_featType = feature_type::LOCALMORANSI; break;
	default:
		throw std::runtime_error("SpidrWrapper: Specified distMetric not supported");
	}

	_SpidrAnalysis = std::make_unique<SpidrAnalysis>();

	_SpidrAnalysis->initializeAnalysisSettings(_featType, _kernelType, _numLocNeighbors, _numHistBins, _aknnAlgType, _distMetric, 0, _numIterations, _perplexity, _exaggeration, _expDecay, _forceCalcBackgroundFeatures);
	_nn = _SpidrAnalysis->getParameters()._nn;

}


void SpidrWrapper::compute_fit(
	py::array_t<float, py::array::c_style | py::array::forcecast> X,
	py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
	int imgWidth, int imgHight,
	std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal) {
	// check input dimensions
	if (X.ndim() != 2)
		throw std::runtime_error("SpidrWrapper: Input should be 2-D NumPy array");

	// copy data from py::array to std::vector
	std::vector<float> dat(X.size());
	std::memcpy(dat.data(), X.data(), X.size() * sizeof(float));

	std::vector<unsigned int> IDs(pointIDsGlobal.size());
	std::memcpy(IDs.data(), pointIDsGlobal.data(), pointIDsGlobal.size() * sizeof(unsigned int));

	// Get other data info
	_numDims = X.shape()[1];
	_numPoints = X.shape()[0];
	_imgSize = ImgSize(imgWidth, imgHight);

	// Pass data to SpidrLib
	if (!backgroundIDsGlobal.has_value())
		_SpidrAnalysis->setupData(dat, IDs, _numDims, _imgSize, "SpidrWrapper");
	else
	{
		std::vector<unsigned int> IDsBack(backgroundIDsGlobal->size());
		std::memcpy(IDsBack.data(), backgroundIDsGlobal->data(), backgroundIDsGlobal->size() * sizeof(unsigned int));
		_SpidrAnalysis->setupData(dat, IDs, _numDims, _imgSize, "SpidrWrapper", IDsBack);
	}

	// Compute knn dists and inds
	_SpidrAnalysis->computeFeatures();
	_SpidrAnalysis->computekNN();

	_fitted = true;

}



std::tuple<std::vector<int>, std::vector<float>> SpidrWrapper::fit(
	py::array_t<float, py::array::c_style | py::array::forcecast> X,
	py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
	int imgWidth, int imgHight,
	std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal) {

	compute_fit(X, pointIDsGlobal, imgWidth, imgHight, backgroundIDsGlobal);

	return _SpidrAnalysis->getKNN();
}


py::array_t<float, py::array::c_style> SpidrWrapper::transform() {

	if (_fitted == false)
		throw std::runtime_error("Call fit() before transform() or go with fit_transform()");

	_SpidrAnalysis->computeEmbedding();
	std::vector<float> emb = _SpidrAnalysis->outputWithBackground();

	return py::array(py::buffer_info(
		emb.data(),													/* data as contiguous array  */
		sizeof(float),												/* size of one scalar        */
		py::format_descriptor<float>::format(),						/* data type                 */
		2,															/* number of dimensions      */
		std::vector<py::ssize_t>{_numPoints, 2},					/* shape of the matrix       */
		std::vector<py::ssize_t>{sizeof(float)*2, sizeof(float)}	/* strides for each axis     */
	));
}


py::array_t<float, py::array::c_style> SpidrWrapper::fit_transform(
	py::array_t<float, py::array::c_style | py::array::forcecast> X,
	py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
	int imgWidth, int imgHight,
	std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal) {

	// fit
	compute_fit(X, pointIDsGlobal, imgWidth, imgHight, backgroundIDsGlobal);

	// transform
	_SpidrAnalysis->computeEmbedding();
	std::vector<float> emb = _SpidrAnalysis->outputWithBackground();

	return py::array(py::buffer_info(
		emb.data(),													/* data as contiguous array  */
		sizeof(float),												/* size of one scalar        */
		py::format_descriptor<float>::format(),						/* data type                 */
		2,															/* number of dimensions      */
		std::vector<py::ssize_t>{_numPoints, 2},					/* shape of the matrix       */
		std::vector<py::ssize_t>{sizeof(float) * 2, sizeof(float)}	/* strides for each axis     */
	));

}

