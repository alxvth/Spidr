#include "FeatureExtraction.h"

#include "KNNUtils.h"
#include "SpidrAnalysisParameters.h"     // class SpidrParameters

#include "hnswlib/hnswlib.h" 
#include "spdlog/spdlog-inl.h"
#include "omp.h"
#include <boost/histogram.hpp>

#include <iterator>     // std::advance
#include <algorithm>    // std::fill, std::find, std::swap_ranges, std::copy, std::set_difference
#include <vector>       // std::vector, std::begin, std::end
#include <numeric>      // std::iota
#include <cmath>        // std::pow
#include <utility>      // std::forward
#include <chrono>       // std::chrono
#include <iterator>		// for ostream_iterator

FeatureExtraction::FeatureExtraction() :
    _neighborhoodSize(1),
    _numHistBins(5),
    _stopFeatureComputation(false),
    _backgroundIDsGlobal(nullptr)
{
    // square neighborhood
    _locNeighbors = ((_neighborhoodSize * 2) + 1) * ((_neighborhoodSize * 2) + 1);
    // uniform weighting
    _neighborhoodWeighting = loc_Neigh_Weighting::WEIGHT_UNIF;
    _neighborhoodWeights.resize(_locNeighbors);
    std::fill(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 1);
}



void FeatureExtraction::compute() {
	spdlog::info("Feature extraction: Started");

	// init, i.e. identify min and max per dimension for histogramming
	initExtraction();

	// all _outFeatures have to be -1 to, so we can easily check later if the were all assigned
	assert(std::all_of(_outFeatures.begin(), _outFeatures.end(), [](float i) {return i == FLT_MAX; }));
	auto start = std::chrono::steady_clock::now();

	// convolution over all points to create histograms
	extractFeatures();

	auto end = std::chrono::steady_clock::now();
	spdlog::info("Feature extraction: Extraction duration (sec): {}", ((float)std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count()) / 1000);

	// if there is a -1 in the _outFeatures, this value was not set at all
    // except if there was background defined, then just go ahead
	assert(!((_backgroundIDsGlobal->empty() || _forceCalcBackgroundFeatures) != std::none_of(_outFeatures.begin(), _outFeatures.end(), [](float i) {return i == FLT_MAX; })));

	spdlog::info("Feature extraction: Finished");
}

void FeatureExtraction::setup(const std::vector<unsigned int>& pointIds, const std::vector<float>& attribute_data, const SpidrParameters& params, std::vector<unsigned int>* backgroundIDsGlobal) {
	spdlog::info("Feature extraction: Setup"); 
	_featType = params._featureType;
    _numFeatureValsPerPoint = params._numFeatureValsPerPoint; 

    // SpidrParameters
    _numHistBins = params._numHistBins;
    _locNeighbors = params._numLocNeighbors;
    _neighborhoodWeighting = params._neighWeighting;

    // Set neighborhood
    _kernelWidth = params._kernelWidth;
    _neighborhoodSize = params._neighborhoodSize;
    weightNeighborhood(_neighborhoodWeighting);     // sets _neighborhoodWeights and _neighborhoodWeightsSum

    // Data
    // Input
    _imgSize = params._imgSize;
    _pointIds = pointIds;
    _numPoints = _pointIds.size();
    _numDims = params._numDims;
    _attribute_data = attribute_data;
    _backgroundIDsGlobal = backgroundIDsGlobal;
    _forceCalcBackgroundFeatures = params._forceCalcBackgroundFeatures;

    if (_backgroundIDsGlobal->empty() && _forceCalcBackgroundFeatures)
        spdlog::warn("Feature extraction: Cannot force to calc features to background if no background is given");

    assert(_attribute_data.size() == _numPoints * _numDims);

    if (_featType == feature_type::TEXTURE_HIST_1D)
    {
        featFunct = &FeatureExtraction::calculateHistogram;  // will be called as calculateHistogram(_pointIds[pointID], neighborValues);
		spdlog::info("Feature extraction: Type 1d texture histogram, Num Bins: {}", _numHistBins);
    }
    else if(_featType == feature_type::LOCALMORANSI)
    {
        featFunct = &FeatureExtraction::calculateLISA;
		spdlog::info("Feature extraction: Local Moran's I");
    }
    else if (_featType == feature_type::LOCALGEARYC)
    {
        featFunct = &FeatureExtraction::calculateGearysC;
        spdlog::info("Feature extraction: Local Geary's C");
    }
    else if (_featType == feature_type::PCLOUD)
    {
        featFunct = &FeatureExtraction::allNeighborhoodIDs; // allNeighborhoodVals for using the data instead of the IDs
		spdlog::info("Feature extraction: Point cloud (just the neighborhood, no transformations)");
    }
    else if (_featType == feature_type::MVN)
    {
        featFunct = &FeatureExtraction::calculateSumAllDist;
        _locNeighbors = 0;      // even better was to skip the neighborhood extraction during
        _neighborhoodSize = 1;  // feature calculation but this is the next best thing
		spdlog::info("Feature extraction: Preparation for Frobenius norm of attribute dist matrices");
    }
    else
    {
        featFunct = NULL;
		spdlog::error("Feature extraction: unknown feature type");
    }

	spdlog::info("Feature extraction: Num neighbors (in each direction): {0} (total neighbors: {1}) Neighbor weighting: {2}", _locNeighbors , _neighborhoodSize, static_cast<unsigned int> (_neighborhoodWeighting));

}

void FeatureExtraction::initExtraction() {
	spdlog::info("Feature extraction: Init feature extraction");

    _outFeatures.resize(_numPoints * _numFeatureValsPerPoint);

    // fill such that _outFeatures are always initialized to FLT_MAX
    std::fill(_outFeatures.begin(), _outFeatures.end(), FLT_MAX);

    // calculate other help values specific to feature type
    if (_featType == feature_type::TEXTURE_HIST_1D) {
        // find min and max for each channel, resize the output larger due to vector features
        _minMaxVals = CalcMinMaxPerChannel(_numPoints, _numDims, _attribute_data);
    }
    else if ((_featType == feature_type::LOCALMORANSI) | (_featType == feature_type::LOCALGEARYC)) {
        // find mean and varaince for each channel
        _meanVals = CalcMeanPerChannel(_numPoints, _numDims, _attribute_data);
        _varVals = CalcVarEstimate(_numPoints, _numDims, _attribute_data, _meanVals);
    }

}

void FeatureExtraction::extractFeatures() {
	spdlog::info("Feature extraction: Extract features");

    // skip if background is given, 
    if (!_backgroundIDsGlobal->empty() && !_forceCalcBackgroundFeatures) {
        std::vector<unsigned int> all_IDs(_numPoints);
        std::vector<unsigned int> foreground_IDs;
        std::iota(all_IDs.begin(), all_IDs.end(), 0);
        std::set_difference(all_IDs.begin(), all_IDs.end(), _backgroundIDsGlobal->begin(), _backgroundIDsGlobal->end(), std::inserter(foreground_IDs, foreground_IDs.begin()));

        // visual studio onlu supports open mp 2.0
#ifdef NDEBUG
#pragma omp parallel for
#endif
        for (int i = 0; i < foreground_IDs.size(); i++) {
            //int pointID = foreground_IDs[i];

            // get neighborhood ids of the current point
            std::vector<int> neighborIDs = neighborhoodIndices(_pointIds[foreground_IDs[i]], _locNeighbors, _imgSize, _pointIds);
            assert(neighborIDs.size() == _neighborhoodSize);

            // get neighborhood values of the current point
            std::vector<float> neighborValues = getNeighborhoodValues(neighborIDs, _attribute_data, _neighborhoodSize, _numDims);
            assert(std::find(neighborValues.begin(), neighborValues.end(), -1) == neighborValues.end());

            // calculate feature(s) for neighborhood
            (this->*featFunct)(_pointIds[foreground_IDs[i]], neighborValues, neighborIDs);  // function pointer defined above

        }
    }
    else
    {
        // convolve over all selected data points
#ifdef NDEBUG
#pragma omp parallel for
#endif
        for (int pointID = 0; pointID < (int)_numPoints; pointID++) {
            // get neighborhood ids of the current point
            std::vector<int> neighborIDs = neighborhoodIndices(_pointIds[pointID], _locNeighbors, _imgSize, _pointIds);
            assert(neighborIDs.size() == _neighborhoodSize);

            // get neighborhood values of the current point
            std::vector<float> neighborValues = getNeighborhoodValues(neighborIDs, _attribute_data, _neighborhoodSize, _numDims);
            assert(std::find(neighborValues.begin(), neighborValues.end(), -1) == neighborValues.end());

            // calculate feature(s) for neighborhood
            (this->*featFunct)(_pointIds[pointID], neighborValues, neighborIDs);  // function pointer defined above
        }
    }
}

void FeatureExtraction::calculateHistogram(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * _numDims * _numHistBins);
    assert(_minMaxVals.size() == 2*_numDims);
    assert(_neighborhoodWeights.size() == _neighborhoodSize);

    float histSum = 0;

    // 1D histograms for each dimension
    for (size_t dim = 0; dim < _numDims; dim++) {
        float minHist = _minMaxVals[2 * dim];
        float maxHist = _minMaxVals[2 * dim + 1];
        if (maxHist == minHist)     // ensure that the histogram can be made
            maxHist += 0.01;

        auto h = boost::histogram::make_histogram(boost::histogram::axis::regular(_numHistBins, minHist, maxHist));
        for (size_t neighbor = 0; neighbor < _neighborhoodSize; neighbor++) {
            if (neighborIDs[neighbor] == -1)
                continue; // skip if neighbor is outside image
            h(neighborValues[neighbor * _numDims + dim], boost::histogram::weight(_neighborhoodWeights[neighbor]));
        }

        assert(h.rank() == 1);                      // 1D hist
        assert(h.axis().size() == _numHistBins);    // right number of bins
        // check if weighting works: sum(hist) == sum(weights) for full spatial neighborhoods
        assert((std::find(neighborIDs.begin(), neighborIDs.end(), -1) == neighborIDs.end()) ? (std::abs(std::accumulate(h.begin(), h.end(), 0.0f) - std::accumulate(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 0.0f)) < 0.01f) : true);

        // normalize the histogram: sum(hist) := 1
        histSum = std::accumulate(h.begin(), h.end(), 0.0f);
        for (auto& hVal : h)
            hVal /= histSum;

        assert(std::abs(std::accumulate(h.begin(), h.end(), 0.0f)-1) < 0.01);    // sum(hist) ~= 1

        // save the histogram in _outFeatures 
        // data layout for points p, dimension d and bin b: [p0d0b0, p0d0b1, p0d0b2, ..., p0d1b0, p0d1b2, ..., p1d0b0, p0d0b1, ...]
        for (size_t bin = 0; bin < _numHistBins; bin++) {
            _outFeatures[pointInd * _numDims * _numHistBins + dim * _numHistBins + bin] = h[bin];
        }

        // values below min are stored in the underflow bin 
        // (they might be below min because they were set to 0 due to being outside the image/selection, i.e. padding reasons)
        if (h.at(-1) != 0) {
            _outFeatures[pointInd * _numDims * _numHistBins + dim * _numHistBins + 0] += h.at(-1);
        }

        // the max value is stored in the overflow bin
        if (h.at(_numHistBins) != 0) {
            _outFeatures[pointInd * _numDims * _numHistBins + dim * _numHistBins + _numHistBins - 1] += h.at(_numHistBins);
        }

    }

}

void FeatureExtraction::calculateLISA(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * _numDims);
    assert(_varVals.size() == _numDims);
    assert(_neighborhoodWeights.size() == _neighborhoodSize);

    float neigh_diff_from_mean_sum = 0;
    float diff_from_mean = 0;
	float local_neighborhoodWeightsSum = 0;

    for (size_t dim = 0; dim < _numDims; dim++) {
        neigh_diff_from_mean_sum = 0;
		local_neighborhoodWeightsSum = 0;
		for (size_t neighbor = 0; neighbor < _neighborhoodSize; neighbor++) {
            if (neighborIDs[neighbor] == -1)
                continue; // skip if neighbor is outside image

			neigh_diff_from_mean_sum += _neighborhoodWeights[neighbor] * (neighborValues[neighbor * _numDims + dim] - _meanVals[dim]);
			local_neighborhoodWeightsSum += _neighborhoodWeights[neighbor];
        }
        diff_from_mean = (_attribute_data[pointInd * _numDims + dim] - _meanVals[dim]);
        // (local_neighborhoodWeightsSum / _varVals[dim]) is the proportionality factor between the local LOCALMORANSI and the global Moran's I
        // such that sum LOCALMORANSI = (local_neighborhoodWeightsSum / _varVals[dim]) * I. Thus, the division by _varVals in the next line yields sum LOCALMORANSI = I. 
        // Cf. 10.1111/j.1538-4632.1995.tb00338.x 
        _outFeatures[pointInd * _numDims + dim] = (diff_from_mean / (local_neighborhoodWeightsSum * _varVals[dim])) * neigh_diff_from_mean_sum;

        // check if local_neighborhoodWeightsSum equals _neighborhoodWeightsSum for full spatial neighborhoods
        assert((std::find(neighborIDs.begin(), neighborIDs.end(), -1) == neighborIDs.end()) ? (local_neighborhoodWeightsSum == _neighborhoodWeightsSum) : true);
    }
}

void FeatureExtraction::calculateGearysC(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * _numDims);
    assert(_meanVals.size() == _numDims);
    assert(_varVals.size() == _numDims);
    assert(_neighborhoodWeights.size() == _neighborhoodSize);

    float diff_from_neigh_sum = 0;
    float diff_from_neigh = 0;
	float local_neighborhoodWeightsSum = 0;

    for (size_t dim = 0; dim < _numDims; dim++) {
        diff_from_neigh_sum = 0;
        diff_from_neigh = 0;
		local_neighborhoodWeightsSum = 0;
		//local_neighborhoodWeightsSum = _neighborhoodWeightsSum;
        for (size_t neighbor = 0; neighbor < _neighborhoodSize; neighbor++) {
            if (neighborIDs[neighbor] == -1)
                continue; // skip if neighbor is outside image

			diff_from_neigh = _attribute_data[pointInd * _numDims + dim] - neighborValues[neighbor * _numDims + dim];
            diff_from_neigh_sum += _neighborhoodWeights[neighbor] * (diff_from_neigh * diff_from_neigh);
			local_neighborhoodWeightsSum += _neighborhoodWeights[neighbor];
        }
        // given that the _neighborhoodWeights sum up to 1, _varVals is the proportionality factor between the local Geary and the global Geary's C
        // such that sum lC = _varVals * gC. Thus, the division by _varVals in the next line yields sum lC = gC. Cf. 10.1111/j.1538-4632.1995.tb00338.x
        _outFeatures[pointInd * _numDims + dim] = ( (2* local_neighborhoodWeightsSum / (_numPoints -1))  / _varVals[dim]) * diff_from_neigh_sum;
    }
}

void FeatureExtraction::calculateSumAllDist(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * (_numDims + 2));

    std::copy_n(_attribute_data.begin() + (pointInd * _numDims), _numDims, _outFeatures.begin() + (pointInd * _numFeatureValsPerPoint));

    int locHeight = std::floor(pointInd / _imgSize.width);         // height val, pixel pos in image
    int locWidth = pointInd - (locHeight * _imgSize.width);        // width val, pixel pos in image

    _outFeatures[pointInd * _numFeatureValsPerPoint + _numDims] = locHeight;
    _outFeatures[pointInd * _numFeatureValsPerPoint + _numDims + 1] = locWidth;
}


void FeatureExtraction::allNeighborhoodVals(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * _numDims * _neighborhoodSize);     // _numFeatureValsPerPoint = _numDims * _neighborhoodSize

    // copy neighborValues into _outFeatures
    std::swap_ranges(neighborValues.begin(), neighborValues.end(), _outFeatures.begin() + (pointInd * _numDims * _neighborhoodSize));
}

void FeatureExtraction::allNeighborhoodIDs(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * _neighborhoodSize);  // _numFeatureValsPerPoint = _neighborhoodSize

    // copy neighborIDs into _outFeatures
    std::replace(neighborIDs.begin(), neighborIDs.end(), -1, -2);       // use -2 mark outsiders, whereas -1 marks not processed
    std::copy(neighborIDs.begin(), neighborIDs.end(), _outFeatures.begin() + (pointInd * _neighborhoodSize));
}

void FeatureExtraction::weightNeighborhood(loc_Neigh_Weighting weighting) {
    _neighborhoodWeights.resize(_neighborhoodSize);

    // Set neighborhood weights
    switch (weighting)
    {
    case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 1); break; 
    case loc_Neigh_Weighting::WEIGHT_BINO: _neighborhoodWeights = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;            // kernel norm: max(_neighborhoodWeights) = 1
    case loc_Neigh_Weighting::WEIGHT_GAUS: _neighborhoodWeights = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_MAX); break;       // kernel norm: max(_neighborhoodWeights) = 1
    default:  std::fill(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), -1);  break;  // no implemented weighting type given. 
    }

    // Some features do not take into account the current point but only the neighborhood values
    // Therefor set the weight of the neighborhood center (the current point) to 0
    if ((_featType == feature_type::LOCALMORANSI) || (_featType == feature_type::LOCALGEARYC)) {
        int centralID = (int)std::sqrt(_neighborhoodSize) + 1;
        assert(_neighborhoodWeights.size() == (centralID-1)*(centralID-1));
        _neighborhoodWeights[centralID] = 0;

        // DEPRECATED normalize neighborhood to the sum w/o the center
        // NormVector(_neighborhoodWeights, std::accumulate(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 0.0f));
    }

    _neighborhoodWeightsSum = std::accumulate(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 0.0f);
}

void FeatureExtraction::setNeighborhoodWeighting(loc_Neigh_Weighting weighting) {
    _neighborhoodWeighting = weighting;
    weightNeighborhood(weighting);
}

void FeatureExtraction::setNumLocNeighbors(size_t size) {
    _locNeighbors = size;
    _kernelWidth = (2 * size) + 1;
    _neighborhoodSize = _kernelWidth * _kernelWidth;
}

void FeatureExtraction::setNumHistBins(size_t size) {
    _numHistBins = size;
}


loc_Neigh_Weighting FeatureExtraction::getNeighborhoodWeighting()
{
    return _neighborhoodWeighting;
}

std::vector<float> FeatureExtraction::output()
{
    return _outFeatures;
}

void FeatureExtraction::stopFeatureCopmutation()
{
    _stopFeatureComputation = false;
}

bool FeatureExtraction::requestedStop()
{
    return _stopFeatureComputation;
}
