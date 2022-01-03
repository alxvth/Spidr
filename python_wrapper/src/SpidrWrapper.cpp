#include "SpidrWrapper.h"
#include "spdlog/spdlog-inl.h"

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
    _perplexity(perplexity), _exaggeration(exaggeration), _expDecay(expDecay), _forceCalcBackgroundFeatures(forceCalcBackgroundFeatures), _fitted(false), _transformed(false)
{
	if (numLocNeighbors <= 0)
		throw std::runtime_error("SpidrWrapper::Constructor: Spatial Neighbors must be larger 0");
	else
		_numLocNeighbors = numLocNeighbors;

	// set _featType depending on distMetric
	switch (_distMetric) {
	case distance_metric::METRIC_QF:
	case distance_metric::METRIC_HEL: 
		_featType = feature_type::TEXTURE_HIST_1D; 
		
		if (_numHistBins <= 0)
			throw std::runtime_error("SpidrWrapper::Constructor: Number of histogram bins must be larger than 0");

		break;
	case distance_metric::METRIC_CHA:
	case distance_metric::METRIC_HAU:
		_featType = feature_type::PCLOUD; break;
	case distance_metric::METRIC_EUC:
		_featType = feature_type::LOCALMORANSI; break;
	case distance_metric::METRIC_BHATTACHARYYA:
		_featType = feature_type::MULTIVAR_NORM; break;
	default:
		throw std::runtime_error("SpidrWrapper::Constructor: Specified distMetric not supported");
	}

	_SpidrAnalysis = std::make_unique<SpidrAnalysis>();
	_nn = static_cast<size_t>(_perplexity) * 3 + 1;  // _perplexity_multiplier = 3

}


void SpidrWrapper::compute_fit(
	py::array_t<float, py::array::c_style | py::array::forcecast> X,
	py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
	int imgWidth, int imgHight,
	std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal) {
	// check input dimensions
	if (X.ndim() != 2)
		throw std::runtime_error("SpidrWrapper::compute_fit: Input should be 2-D NumPy array");

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

	// Init all settings (setupData must have been called before initing the settings.)
	_SpidrAnalysis->initializeAnalysisSettings(_featType, _kernelType, _numLocNeighbors, _numHistBins, _aknnAlgType, _distMetric, _numIterations, _perplexity, _exaggeration, _expDecay, _forceCalcBackgroundFeatures);

	// Compute knn dists and inds
	_SpidrAnalysis->computeFeatures();
	_SpidrAnalysis->computekNN();

	_fitted = true;
	_transformed = false;
}


std::tuple<std::vector<int>, std::vector<float>> SpidrWrapper::fit(
	py::array_t<float, py::array::c_style | py::array::forcecast> X,
	py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
	int imgWidth, int imgHight,
	std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal) {

	// Init settings, (Extract features), compute similarities, embed data
	compute_fit(X, pointIDsGlobal, imgWidth, imgHight, backgroundIDsGlobal);

	return _SpidrAnalysis->getKnn();
}


void SpidrWrapper::compute_transform() {
	if (_fitted == false) {
		spdlog::error("SpidrWrapper::compute_transform: Call fit(...) before transform() or go with fit_transform() or set knn manually with set_kNN(...)");
		return;
	}

	// computes t-SNE based on previously computed high-dimensional distances
	_SpidrAnalysis->computeEmbedding();

	_transformed = true;
}


py::array_t<float, py::array::c_style> SpidrWrapper::transform() {

	// compute embedding
	compute_transform();

	// return embedding
	return get_embedding();
}


py::array_t<float, py::array::c_style> SpidrWrapper::fit_transform(
	py::array_t<float, py::array::c_style | py::array::forcecast> X,
	py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
	int imgWidth, int imgHight,
	std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal) {

	// Init settings, (Extract features), compute similarities
	compute_fit(X, pointIDsGlobal, imgWidth, imgHight, backgroundIDsGlobal);

	// embed data
	compute_transform();

	// return embedding
	return get_embedding();
}

void SpidrWrapper::set_kNN(py::array_t<int, py::array::c_style | py::array::forcecast> knn_indices, py::array_t<float, py::array::c_style | py::array::forcecast> knn_distances) {
	// copy data from py::array to std::vector
	std::vector<int> indices(knn_indices.size());
	std::memcpy(indices.data(), knn_indices.data(), knn_indices.size() * sizeof(float));
	
	std::vector<float> distances(knn_distances.size());
	std::memcpy(distances.data(), knn_distances.data(), knn_distances.size() * sizeof(float));

	// check values
	if (indices.size() != distances.size())
	{
		spdlog::error("SpidrWrapper::setKNN: knn indices and distances do not align.");
		return;
	}

	if (indices.size() % _nn != 0)
	{
		spdlog::error("SpidrWrapper::setKNN: size of indices vector must be multiple of number of neighbors.");
		return;
	}

	// set knn values
	_SpidrAnalysis->initializeAnalysisSettings(_featType, _kernelType, _numLocNeighbors, _numHistBins, _aknnAlgType, _distMetric, _numIterations, _perplexity, _exaggeration, _expDecay, _forceCalcBackgroundFeatures);
	_SpidrAnalysis->setKnn(indices, distances);

	// set number of points as it is used in transform()
	_numPoints = indices.size() / _nn;

	_fitted = true;
	_transformed = false;
}

py::array_t<float, py::array::c_style> SpidrWrapper::get_embedding() {
	if (_transformed == false) {
		spdlog::error("SpidrWrapper::get_embedding: Call compute_transform() or fit_transform() first");
		return py::array_t<float>();
	}

	// get embedding
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

std::tuple<std::vector<int>, std::vector<float>> SpidrWrapper::get_kNN()
{
	if (_fitted == false) {
		spdlog::error("SpidrWrapper::get_kNN: Call fit(...) before transform() or go with fit_transform() or set knn manually with set_kNN(...)");
		return std::make_tuple(std::vector<int>{ -1 }, std::vector<float>{ 0 });	// return dummy values
	}

	return _SpidrAnalysis->getKnn();
}