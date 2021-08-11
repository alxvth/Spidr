#include "FeatureExtraction.h"

#include "KNNUtils.h"
#include "SpidrAnalysisParameters.h"     // class SpidrParameters

#include "hnswlib/hnswlib.h" 
#include "spdlog/spdlog-inl.h"
#include "omp.h"

#include <iterator>     // std::advance
#include <algorithm>    // std::fill, std::find, std::swap_ranges, std::copy, std::set_difference
#include <vector>       
#include <array>       
#include <numeric>      // std::iota
#include <cmath>        // std::pow
#include <utility>      // std::forward
#include <chrono>       // std::chrono
#include <iostream> 

FeatureExtraction::FeatureExtraction() :
    _neighborhoodSize(1),
    _numHistBins(5),
    _stopFeatureComputation(false)
{
    // square neighborhood
    _numLocNeighbors = ((_neighborhoodSize * 2) + 1) * ((_neighborhoodSize * 2) + 1);
    // uniform weighting
    _neighborhoodWeighting = loc_Neigh_Weighting::WEIGHT_UNIF;
    _neighborhoodWeights.resize(_numLocNeighbors);
    std::fill(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 1);
}



void FeatureExtraction::compute() {
	spdlog::info("Feature extraction: Started");

	// init, i.e. identify min and max per dimension
	initExtraction();

	// all _outFeatures have to be FLT_MAX to, so we can easily check later if the were all assigned
	assert(std::all_of(_outFeatures.begin(), _outFeatures.end(), [](float i) {return i == FLT_MAX; }));
	auto start = std::chrono::steady_clock::now();

	// for each points, compute the features for the respective neighborhood
	extractFeatures();

	auto end = std::chrono::steady_clock::now();
	spdlog::info("Feature extraction: Extraction duration (sec): {}", ((float)std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count()) / 1000);

    // No value in _outFeatures should be FLT_MAX (it's init value)
    // Except when background IDs are given: then they were skipped during the feature computation and still have their initial FLT_MAX value
    // The background entries will be ignored during the Distance calculation 
	assert(!((_backgroundIDsGlobal.empty() || _forceCalcBackgroundFeatures) != std::none_of(_outFeatures.begin(), _outFeatures.end(), [](float i) {return i == FLT_MAX; })));

	spdlog::info("Feature extraction: Finished");
}

void FeatureExtraction::setup(const std::vector<unsigned int>& pointIDsGlobal, const std::vector<float>& attribute_data, const SpidrParameters& params, 
                              const std::vector<unsigned int>& backgroundIDsGlobal, const std::vector<unsigned int>& foregroundIDsGlobal) {
	spdlog::info("Feature extraction: Setup"); 
	_featType = params._featureType;
    _numFeatureValsPerPoint = params._numFeatureValsPerPoint; 

    // SpidrParameters
    _numHistBins = params._numHistBins;
    _numLocNeighbors = params._numLocNeighbors;
    _neighborhoodWeighting = params._neighWeighting;

    // Set neighborhood
    _kernelWidth = params._kernelWidth;
    _neighborhoodSize = params._neighborhoodSize;
    weightNeighborhood(_neighborhoodWeighting);     // sets _neighborhoodWeights and _neighborhoodWeightsSum

    // Data
    // Input
    _imgSize = params._imgSize;
    _pointIDsGlobal = pointIDsGlobal;
    _numPoints = _pointIDsGlobal.size();
    _numDims = params._numDims;
    _attribute_data = attribute_data;
    _backgroundIDsGlobal = backgroundIDsGlobal;
    _foregroundIDsGlobal = foregroundIDsGlobal;
    _forceCalcBackgroundFeatures = params._forceCalcBackgroundFeatures;

    if (_backgroundIDsGlobal.empty() && _forceCalcBackgroundFeatures)
        spdlog::warn("Feature extraction: Cannot force to calc features to background if no background is given");

    // Convert the background IDs into an Eigen matrix
    // there is no standard Eigen typedef for unsigned typesa and Eigen::MatrixXi does not work
    Eigen::MatrixXui _indices_mat = Eigen::Map<Eigen::MatrixXui>(&_pointIDsGlobal[0], _imgSize.width, _imgSize.height);
    // pad the matrix in all directions with _numLocNeighbors values
    _indices_mat_padded = padConst(_indices_mat, _numLocNeighbors);

    assert(_attribute_data.size() == _numPoints * _numDims);

    if (_featType == feature_type::TEXTURE_HIST_1D)
    {
        featFunct = &FeatureExtraction::calculateHistogram;  // will be called as calculateHistogram(_pointIDsGlobal[pointID], neighborValues);
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
    else if (_featType == feature_type::MULTIVAR_NORM)
    {
        featFunct = &FeatureExtraction::multivarNormDistDescriptor;
        spdlog::info("Feature extraction: Multivariate normal distribution descriptors (covaraince matrix and channel-wise mean)");
    }
    else
    {
        featFunct = NULL;
		spdlog::error("Feature extraction: unknown feature type");
    }

	spdlog::info("Feature extraction: Num neighbors (in each direction): {0} (total neighbors: {1}) Neighbor weighting: {2}", _numLocNeighbors , _neighborhoodSize, static_cast<unsigned int> (_neighborhoodWeighting));

}

void FeatureExtraction::initExtraction() {
	spdlog::info("Feature extraction: Init feature extraction");
    _outFeatures.resize(_numPoints * _numFeatureValsPerPoint);
    _outFeaturesF.resize(_numPoints);

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

    std::vector<unsigned int>* IDs;

    // Only calc features for foreground, execpt when _forceCalcBackgroundFeatures is set and a background is given
    if ((_backgroundIDsGlobal.empty() != false) && _forceCalcBackgroundFeatures)
        IDs = &_pointIDsGlobal;
    else
        IDs = &_foregroundIDsGlobal;

    // Iterate over IDs and compute features
#ifdef NDEBUG
#pragma omp parallel for
#endif
    for (int i = 0; i < IDs->size(); i++) {

        // get neighborhood ids of the current point
        std::vector<int> neighborIDs = getNeighborhoodInds((*IDs)[i] - ((*IDs)[i] / _imgSize.width) * _imgSize.width, (*IDs)[i] / _imgSize.width, _kernelWidth, &_indices_mat_padded);
        assert(neighborIDs.size() == _neighborhoodSize);

        // get neighborhood values of the current point
        std::vector<float> neighborValues = getNeighborhoodValues(neighborIDs, _attribute_data, _neighborhoodSize, _numDims);
        assert(std::none_of(neighborValues.begin(), neighborValues.end(), [](float neighborVal) { return neighborVal == FLT_MAX; })); // check no value is FLT_MAX, which would indicate an unset value

        // calculate feature(s) for neighborhood
        (this->*featFunct)((*IDs)[i], neighborValues, neighborIDs);  // function pointer defined above

    }
}

void FeatureExtraction::calculateHistogram(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * _numDims * _numHistBins);
    assert(_minMaxVals.size() == 2*_numDims);
    assert(_neighborhoodWeights.size() == _neighborhoodSize);
    assert(std::none_of(neighborIDs.begin(), neighborIDs.end(), [](int i) {return i == -1; }));

    Eigen::VectorXf normHist;
    std::vector<Eigen::VectorXf> feat(_numDims);

    // 1D histograms for each dimension
    for (size_t dim = 0; dim < _numDims; dim++) {
        float minHist = _minMaxVals[2 * dim];
        float maxHist = _minMaxVals[2 * dim + 1];
        if (maxHist == minHist)     // ensure that the histogram can be made
            maxHist += 0.01;

        Histogram_Weighted hist = Histogram_Weighted(minHist, maxHist, static_cast<unsigned int>(_numHistBins)); 
        for (size_t neighbor = 0; neighbor < _neighborhoodSize; neighbor++) {
            hist.fill_weighted(neighborValues[neighbor * _numDims + dim], _neighborhoodWeights[neighbor]);
        }

        assert(hist.getCountUnderflow() == 0);
        assert(hist.getCountOverflow() == 0);

        // check if weighting works: sum(hist) == sum(weights) for full spatial neighborhoods
        assert(std::abs(std::accumulate(hist.cbegin(), hist.cend(), 0.0f) - std::accumulate(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 0.0f)) < 0.01f);

        // normalize the histogram: sum(hist) := 1
        normHist = hist.normalizedCounts();
        assert(std::abs(normHist.sum() - 1) < 0.01);  

        // save the histogram in _outFeatures 
        // data layout for points p, dimension d and bin b: [p0d0b0, p0d0b1, p0d0b2, ..., p0d1b0, p0d1b2, ..., p1d0b0, p0d0b1, ...]
        for(size_t bin = 0; bin < _numHistBins; bin++)
            _outFeatures[pointInd * _numDims * _numHistBins + dim * _numHistBins + bin] = normHist[bin];

        feat[dim] = normHist;

    }
    _outFeaturesF.get_data_ptr()->at(pointInd) = new FeatureData<std::vector<Eigen::VectorXf>>(feat);

}

void FeatureExtraction::calculateLISA(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * _numDims);
    assert(_varVals.size() == _numDims);
    assert(_neighborhoodWeights.size() == _neighborhoodSize);

    float neigh_diff_from_mean_sum = 0;
    float diff_from_mean = 0;
	float local_neighborhoodWeightsSum = 0;

    std::vector<float> feat(_numDims);

    for (size_t dim = 0; dim < _numDims; dim++) {
        neigh_diff_from_mean_sum = 0;
		local_neighborhoodWeightsSum = 0;
		for (size_t neighbor = 0; neighbor < _neighborhoodSize; neighbor++) {
			neigh_diff_from_mean_sum += _neighborhoodWeights[neighbor] * (neighborValues[neighbor * _numDims + dim] - _meanVals[dim]);
			local_neighborhoodWeightsSum += _neighborhoodWeights[neighbor];
        }
        diff_from_mean = (_attribute_data[pointInd * _numDims + dim] - _meanVals[dim]);
        // (local_neighborhoodWeightsSum / _varVals[dim]) is the proportionality factor between the local LOCALMORANSI and the global Moran's I
        // such that sum LOCALMORANSI = (local_neighborhoodWeightsSum / _varVals[dim]) * I. Thus, the division by _varVals in the next line yields sum LOCALMORANSI = I. 
        // Cf. 10.1111/j.1538-4632.1995.tb00338.x 
        _outFeatures[pointInd * _numDims + dim] = (diff_from_mean / (local_neighborhoodWeightsSum * _varVals[dim])) * neigh_diff_from_mean_sum;
        feat[dim] = (diff_from_mean / (local_neighborhoodWeightsSum * _varVals[dim])) * neigh_diff_from_mean_sum;

        // check if local_neighborhoodWeightsSum equals _neighborhoodWeightsSum for full spatial neighborhoods
        assert((std::find(neighborIDs.begin(), neighborIDs.end(), -1) == neighborIDs.end()) ? (local_neighborhoodWeightsSum == _neighborhoodWeightsSum) : true);
    }

    _outFeaturesF.get_data_ptr()->at(pointInd) = new FeatureData<std::vector<float>>(feat);

}

void FeatureExtraction::calculateGearysC(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * _numDims);
    assert(_meanVals.size() == _numDims);
    assert(_varVals.size() == _numDims);
    assert(_neighborhoodWeights.size() == _neighborhoodSize);

    float diff_from_neigh_sum = 0;
    float diff_from_neigh = 0;
	float local_neighborhoodWeightsSum = 0;

    std::vector<float> feat(_numDims);

    for (size_t dim = 0; dim < _numDims; dim++) {
        diff_from_neigh_sum = 0;
        diff_from_neigh = 0;
		local_neighborhoodWeightsSum = 0;
		//local_neighborhoodWeightsSum = _neighborhoodWeightsSum;
        for (size_t neighbor = 0; neighbor < _neighborhoodSize; neighbor++) {
			diff_from_neigh = _attribute_data[pointInd * _numDims + dim] - neighborValues[neighbor * _numDims + dim];
            diff_from_neigh_sum += _neighborhoodWeights[neighbor] * (diff_from_neigh * diff_from_neigh);
			local_neighborhoodWeightsSum += _neighborhoodWeights[neighbor];
        }
        // given that the _neighborhoodWeights sum up to 1, _varVals is the proportionality factor between the local Geary and the global Geary's C
        // such that sum lC = _varVals * gC. Thus, the division by _varVals in the next line yields sum lC = gC. Cf. 10.1111/j.1538-4632.1995.tb00338.x
        _outFeatures[pointInd * _numDims + dim] = ( (2* local_neighborhoodWeightsSum / (_numPoints -1))  / _varVals[dim]) * diff_from_neigh_sum;
        feat[dim] =                               ( (2 *local_neighborhoodWeightsSum / (_numPoints - 1)) / _varVals[dim]) * diff_from_neigh_sum;
    }

    _outFeaturesF.get_data_ptr()->at(pointInd) = new FeatureData<std::vector<float>>(feat);

}

void FeatureExtraction::calculateSumAllDist(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * (_numDims + 2));

    std::copy_n(_attribute_data.begin() + (pointInd * _numDims), _numDims, _outFeatures.begin() + (pointInd * _numFeatureValsPerPoint));

    int locHeight = std::floor(pointInd / _imgSize.width);         // height val, pixel pos in image
    int locWidth = pointInd - (locHeight * _imgSize.width);        // width val, pixel pos in image

    std::array<int, 2> feat{ locHeight, locWidth};
    _outFeaturesF.get_data_ptr()->at(pointInd) = new FeatureData<std::array<int, 2>>(feat);

    _outFeatures[pointInd * _numFeatureValsPerPoint + _numDims] = locHeight;
    _outFeatures[pointInd * _numFeatureValsPerPoint + _numDims + 1] = locWidth;
}


void FeatureExtraction::multivarNormDistDescriptor(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * (_numDims + _numDims * _numDims));
    assert(_neighborhoodWeights.size() == _neighborhoodSize);

    // transform std data to eigen
    Eigen::MatrixXf neighborValues_mat(_numDims, _neighborhoodSize);
    for (int ch = 0; ch < _numDims; ch++)
        neighborValues_mat.row(ch) = Eigen::Map<Eigen::VectorXf>(neighborValues.data() + ch* _neighborhoodSize, _neighborhoodSize);

    // compute features
    multivar_normal mean_covmat = compMultiVarFeatures(neighborValues_mat, _neighborhoodWeights_eig);

    // save features
    multivar_normal_plusDet feat = multivar_normal_plusDet(mean_covmat.first, mean_covmat.second, mean_covmat.second.determinant());
    std::vector<IFeatureData*>* ib_featdata = _outFeaturesF.get_data_ptr();
    ib_featdata->at(pointInd) = new FeatureData<multivar_normal_plusDet> (feat);

    // transform features back to std and save
    std::swap_ranges(mean_covmat.first.begin(), mean_covmat.first.end(), _outFeatures.begin() + (pointInd * _numFeatureValsPerPoint));
    for (int ch = 0; ch < _numDims; ch++) // swap row wise because straightforward begin to end range swap did not work...
        std::swap_ranges(mean_covmat.second.row(ch).begin(), mean_covmat.second.row(ch).end(), _outFeatures.begin() + (pointInd * _numFeatureValsPerPoint + _numDims * (ch +1) ));

}


void FeatureExtraction::allNeighborhoodVals(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * _numDims * _neighborhoodSize);     // _numFeatureValsPerPoint = _numDims * _neighborhoodSize

    // copy neighborValues into _outFeatures
    std::swap_ranges(neighborValues.begin(), neighborValues.end(), _outFeatures.begin() + (pointInd * _numDims * _neighborhoodSize));

    _outFeaturesF.get_data_ptr()->at(pointInd) = new FeatureData<std::vector<float>>(neighborValues);

}

void FeatureExtraction::allNeighborhoodIDs(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_outFeatures.size() == _numPoints * _neighborhoodSize);  // _numFeatureValsPerPoint = _neighborhoodSize

    // copy neighborIDs into _outFeatures
    //std::replace(neighborIDs.begin(), neighborIDs.end(), -1, -2);       // use -2 mark outsiders, whereas -1 marks not processed
    std::copy(neighborIDs.begin(), neighborIDs.end(), _outFeatures.begin() + (pointInd * _neighborhoodSize));
    //std::swap_ranges(neighborIDs.begin(), neighborIDs.end(), _outFeatures.begin() + (pointInd * _neighborhoodSize));

    _outFeaturesF.get_data_ptr()->at(pointInd) = new FeatureData<std::vector<int>>(neighborIDs);

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

    _neighborhoodWeights_eig = Eigen::Map<Eigen::VectorXf>(_neighborhoodWeights.data(), _neighborhoodSize);
    assert(_neighborhoodWeights_eig.sum() == _neighborhoodWeightsSum);
}

void FeatureExtraction::setNeighborhoodWeighting(loc_Neigh_Weighting weighting) {
    _neighborhoodWeighting = weighting;
    weightNeighborhood(weighting);
}

void FeatureExtraction::setNumLocNeighbors(size_t size) {
    _numLocNeighbors = size;
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

Feature FeatureExtraction::outputF()
{
    return _outFeaturesF;
}


void FeatureExtraction::stopFeatureCopmutation()
{
    _stopFeatureComputation = false;
}

bool FeatureExtraction::requestedStop()
{
    return _stopFeatureComputation;
}
