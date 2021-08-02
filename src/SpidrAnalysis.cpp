#include "SpidrAnalysis.h"

#include <cmath>
#include <algorithm>
#include <chrono>       // std::chrono
#include "spdlog/spdlog-inl.h"


SpidrAnalysis::SpidrAnalysis() {};

void SpidrAnalysis::setupData(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, const size_t numDimensions, const ImgSize imgSize, const std::string embeddingName, const std::vector<unsigned int>& backgroundIDsGlobal) {
    // TODO: there should be a function that calls setupData and initializeAnalysisSettings so that the user only sees a single setup function.

	// Set data
    _attribute_data = attribute_data;
    _pointIDsGlobal = pointIDsGlobal;
    _backgroundIDsGlobal = backgroundIDsGlobal;
    std::sort(_backgroundIDsGlobal.begin(), _backgroundIDsGlobal.end());
    // IDs that are not background are in the foreground
    std::set_difference(_pointIDsGlobal.begin(), _pointIDsGlobal.end(),
                        _backgroundIDsGlobal.begin(), _backgroundIDsGlobal.end(), 
                        std::inserter(_foregroundIDsGlobal, _foregroundIDsGlobal.begin()));

    // Set parameters
    _params._numPoints = _pointIDsGlobal.size();
    _params._numDims = numDimensions;
	_params._imgSize = imgSize;
    _params._embeddingName = embeddingName;
    _params._dataVecBegin = _attribute_data.data();          // used in point cloud distance

	spdlog::info("SpidrAnalysis: Setup data with number of points: {0}, num dims: {1}, image size (width, height): {2}", _params._numPoints, _params._numDims, _params._imgSize.width, _params._imgSize.height);
    if(!_backgroundIDsGlobal.empty())
        spdlog::info("SpidrAnalysis: Excluding {} background points and respective features", _backgroundIDsGlobal.size());

}

void SpidrAnalysis::initializeAnalysisSettings(const feature_type featType, const loc_Neigh_Weighting kernelWeightType, const size_t numLocNeighbors, const size_t numHistBins,\
                                               const knn_library aknnAlgType, const distance_metric aknnMetric, const float MVNweight, \
                                               const int numIterations, const int perplexity, const int exaggeration, const int expDecay, bool forceCalcBackgroundFeatures) {
	if (_params._numDims < 0 || _params._numHistBins < 0)
		spdlog::error("SpidrWrapper: first call SpidrAnalysis::setupData() before initializing the settings with SpidrAnalysis::initializeAnalysisSettings since some might depend on the data dimensions.");

	// the following set* functions set values in _params
	// initialize Feature Extraction Settings
    setFeatureType(featType);
    setKernelWeight(kernelWeightType);
    setNumLocNeighbors(numLocNeighbors);    // Sets both _params._kernelWidth and _params._neighborhoodSize
    setNumHistBins(numHistBins);

    // initialize Distance Calculation Settings
    // number of nn is dertermined by perplexity, set in setPerplexity
    setKnnAlgorithm(aknnAlgType);
    setDistanceMetric(aknnMetric);
    setMVNWeight(MVNweight);

    // Initialize the tSNE computation
    setNumIterations(numIterations);
    setPerplexity(perplexity);
    setExaggeration(exaggeration);
    setExpDecay(expDecay);

    // Derived parameters
	setNumFeatureValsPerPoint(featType, _params._numDims, _params._numHistBins, _params._neighborhoodSize);			// sets _params._numFeatureValsPerPoint
    setForceCalcBackgroundFeatures(forceCalcBackgroundFeatures);													// sets _params._forceCalcBackgroundFeatures

	spdlog::info("SpidrAnalysis: Initialized all settings");
}


void SpidrAnalysis::computeFeatures() {
	_featExtraction.setup(_pointIDsGlobal, _attribute_data, _params, &_backgroundIDsGlobal);
	_featExtraction.compute();
	spdlog::info("SpidrAnalysis: Get computed feature values");
	_dataFeats = _featExtraction.output();

}

void SpidrAnalysis::computekNN() {
	_distCalc.setup(_dataFeats, _backgroundIDsGlobal, _params);
	_distCalc.compute();
	_knn_indices = _distCalc.get_knn_indices();
	_knn_distances_squared = _distCalc.get_knn_distances_squared();
}

void SpidrAnalysis::computeEmbedding() {
	_tsne.setup(_knn_indices, _knn_distances_squared, _params);
	_tsne.compute();
}

void SpidrAnalysis::compute() {
	// Extract features
	computeFeatures();

    // Caclculate distances and kNN
	computekNN();

    // Compute t-SNE with the given data
	computeEmbedding();

	spdlog::info("SpidrAnalysis: Finished");
}


void SpidrAnalysis::setFeatureType(const feature_type feature_type) {
	_params._featureType = feature_type;
}

void SpidrAnalysis::setKernelWeight(const loc_Neigh_Weighting loc_Neigh_Weighting) {
    _params._neighWeighting = loc_Neigh_Weighting;
}

void SpidrAnalysis::setNumLocNeighbors(const size_t num) {
    _params._numLocNeighbors = num;
    _params._kernelWidth = (2 * _params._numLocNeighbors) + 1;
    _params._neighborhoodSize = _params._kernelWidth * _params._kernelWidth;;
}

void SpidrAnalysis::setNumHistBins(const size_t num) {
    _params._numHistBins = num;
}

void SpidrAnalysis::setKnnAlgorithm(const knn_library knn_library) {
    _params._aknn_algorithm = knn_library;
}

void SpidrAnalysis::setDistanceMetric(const distance_metric distance_metric) {
    _params._aknn_metric = distance_metric;
}

void SpidrAnalysis::setPerplexity(const unsigned perplexity) {
    _params._perplexity = perplexity;
    _params._nn = (perplexity * _params._perplexity_multiplier) + 1;    // see Van Der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms. The Journal of Machine Learning Research, 15(1), 3221-3245.

    // For small images, use less kNN
    if (_params._nn > _params._numPoints)
        _params._nn = _params._numPoints;
}

void SpidrAnalysis::setNumIterations(const unsigned numIt) {
    _params._numIterations = numIt;
}

void SpidrAnalysis::setExaggeration(const unsigned exag) {
    _params._exaggeration = exag;
}

void SpidrAnalysis::setExpDecay(const unsigned expDecay) {
    _params._expDecay = expDecay;
}

void SpidrAnalysis::setNumFeatureValsPerPoint(feature_type featType, size_t numDims, size_t numHistBins, size_t neighborhoodSize) {
	_params._numFeatureValsPerPoint = NumFeatureValsPerPoint(featType, numDims, numHistBins, neighborhoodSize);
}

void SpidrAnalysis::setMVNWeight(const float weight) {
    _params._MVNweight = weight;
}

void SpidrAnalysis::setForceCalcBackgroundFeatures(const bool CalcBackgroundFeatures) {
    _params._forceCalcBackgroundFeatures = CalcBackgroundFeatures;
}

const size_t SpidrAnalysis::getNumEmbPoints() {
    return _params._numPoints;
}

const size_t SpidrAnalysis::getNumImagePoints() {
    assert(_pointIDsGlobal.size() == _params._numPoints + _backgroundIDsGlobal.size());
    return _pointIDsGlobal.size();
}

bool SpidrAnalysis::embeddingIsRunning() {
    return _tsne.isTsneRunning();
}

const std::vector<float>& SpidrAnalysis::output() {
    return _tsne.output();
}

const std::vector<float>& SpidrAnalysis::outputWithBackground() {
    const std::vector<float>& emb = _tsne.output();

    if (_backgroundIDsGlobal.empty())
    {
        return emb;
    }
    else
    {
        addBackgroundToEmbedding(_emd_with_backgound, emb);
        return _emd_with_backgound;
    }
}

void SpidrAnalysis::addBackgroundToEmbedding(std::vector<float>& emb, const std::vector<float>& emb_wo_bg) {
    spdlog::info("SpidrAnalysis: Add background back to embedding");
    auto start = std::chrono::steady_clock::now();

    emb.resize(_pointIDsGlobal.size() * 2);

    // find min x and min y embedding positions
    float minx = emb_wo_bg[0];
    float miny = emb_wo_bg[1];

    for (size_t i = 0; i < emb_wo_bg.size(); i += 2) {
        if (emb_wo_bg[i] < minx)
            minx = emb_wo_bg[i];

        if (emb_wo_bg[i + 1] < miny)
            miny = emb_wo_bg[i + 1];
    }

    minx -= std::abs(minx) * 0.05;
    miny -= std::abs(miny) * 0.05;

    // Place all background pixel in the lower left corner of the embedding
#ifdef NDEBUG
#pragma omp parallel for
#endif
    for (int i = 0; i < _backgroundIDsGlobal.size(); i++) {
        emb[2 * _backgroundIDsGlobal[i]] = minx;
        emb[2 * _backgroundIDsGlobal[i] + 1] = miny;

    }

    // Copy the foreground embedding positions
#ifdef NDEBUG
#pragma omp parallel for
#endif
    for (int i = 0; i < _foregroundIDsGlobal.size(); i++) {
        emb[2 * _foregroundIDsGlobal[i]] = emb_wo_bg[2 * i];
        emb[2 * _foregroundIDsGlobal[i] + 1] = emb_wo_bg[2 * i + 1];
    }

    auto end = std::chrono::steady_clock::now();
    spdlog::info("SpidrAnalysis: Add backgorund (sec): {}", ((float)std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count()) / 1000);


}

void SpidrAnalysis::stopComputation() {
    _featExtraction.stopFeatureCopmutation();
    _tsne.stopGradientDescent();
}

const SpidrParameters SpidrAnalysis::getParameters() {
    return _params;
}

const std::vector<float> SpidrAnalysis::getDataFeatures() {
	return _dataFeats;
}

const std::tuple<std::vector<int>, std::vector<float>> SpidrAnalysis::getKNN() {
	return std::make_tuple(_knn_indices, _knn_distances_squared);
}
