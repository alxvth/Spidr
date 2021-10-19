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
    auto start = std::chrono::steady_clock::now();

	// init, i.e. identify min and max per dimension
	initExtraction();

	// for each points, compute the features for the respective neighborhood
	extractFeatures();

	auto end = std::chrono::steady_clock::now();
	spdlog::info("Feature extraction: Extraction duration (sec): {}", ((float)std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count()) / 1000);
	spdlog::info("Feature extraction: Finished");
}

void FeatureExtraction::setup(const std::vector<unsigned int>& pointIDsGlobal, const std::vector<float>& attribute_data, const SpidrParameters& params, 
                              const std::vector<unsigned int>& backgroundIDsGlobal, const std::vector<unsigned int>& foregroundIDsGlobal) {
	spdlog::info("Feature extraction: Setup"); 
	_featType = params._featureType;
    _distType = params._aknn_metric;
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
    // pad the matrix in all directions with the pad with _numLocNeighbors with the edge (border value)
    _indices_mat_padded = padEdge(_indices_mat, _numLocNeighbors);

    assert(_attribute_data.size() == _numPoints * _numDims);

    if (_featType == feature_type::TEXTURE_HIST_1D)
    {
        featFunct = &FeatureExtraction::calculateHistogram;  // will be called as calculateHistogram(_pointIDsGlobal[pointID], neighborValues);
		spdlog::info("Feature extraction: Type 1d texture histogram, Num Bins: {}", _numHistBins);
    }
    else if (_featType == feature_type::CHANNEL_HIST)
    {
        featFunct = &FeatureExtraction::calculateChannelHistogram;
        spdlog::info("Feature extraction: Channel Histograms, i.e. one bin per channel");
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
        featFunct = &FeatureExtraction::allNeighborhoodVals;  // allNeighborhoodIDs OR allNeighborhoodVals
		spdlog::info("Feature extraction: Point cloud (all neighborhood values, no transformations)");
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

	spdlog::info("Feature extraction: Num neighbors (in each direction): {0} (total neighbors: {1}) Neighbor weighting: {2}", _numLocNeighbors , _neighborhoodSize, logging::neighborhood_weighting_name(_neighborhoodWeighting));

}

void FeatureExtraction::initExtraction() {
	spdlog::info("Feature extraction: Init feature extraction");
    _outFeatures.resize(_numPoints);

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
        feat[dim] = normHist;
    }
    _outFeatures.get_data_ptr()->at(pointInd) = new FeatureData<std::vector<Eigen::VectorXf>>(feat);

}

void FeatureExtraction::calculateChannelHistogram(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(std::none_of(neighborIDs.begin(), neighborIDs.end(), [](int i) {return i == -1; }));

    Channel_Histogram_Weighted channelHist = Channel_Histogram_Weighted(_numDims);

    for (unsigned int neighID = 0; neighID < _neighborhoodSize; neighID++) {
        {
            for (unsigned int dim = 0; dim < _numDims; dim++)
                channelHist.fill_ch_weighted(dim, neighborValues[neighID * _numDims + dim], _neighborhoodWeights[neighID]);
        }
    }

    _outFeatures.get_data_ptr()->at(pointInd) = new FeatureData<std::vector<float>>(channelHist.counts_std());
}

void FeatureExtraction::calculateLISA(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
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
        feat[dim] = (diff_from_mean / (local_neighborhoodWeightsSum * _varVals[dim])) * neigh_diff_from_mean_sum;

        // check if local_neighborhoodWeightsSum equals _neighborhoodWeightsSum for full spatial neighborhoods
        assert((std::find(neighborIDs.begin(), neighborIDs.end(), -1) == neighborIDs.end()) ? (local_neighborhoodWeightsSum == _neighborhoodWeightsSum) : true);
    }

    _outFeatures.get_data_ptr()->at(pointInd) = new FeatureData<std::vector<float>>(feat);

}

void FeatureExtraction::calculateGearysC(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
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
        feat[dim] = ( (2 *local_neighborhoodWeightsSum / (_numPoints - 1)) / _varVals[dim]) * diff_from_neigh_sum;
    }

    _outFeatures.get_data_ptr()->at(pointInd) = new FeatureData<std::vector<float>>(feat);

}


void FeatureExtraction::multivarNormDistDescriptor(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    assert(_neighborhoodWeights.size() == _neighborhoodSize);

    // transform std data to eigen
    // data layout of neighborValues with dimension d and neighbor n: [n0d0, n0d1, n0d2, ..., n1d0, n1d1, ..., n2d0, n2d1, ...]
    Eigen::MatrixXf neighborValues_mat(_numDims, _neighborhoodSize);
    for (int d = 0; d < _numDims; d++) {
        for (int n = 0; n < _neighborhoodSize; n++)
        {
            neighborValues_mat.row(d)[n] = neighborValues[n * _numDims + d];
        }
    }

    // compute features: mean vector and covariance matrix
    Multivar_normal meanCov_feat = compMultiVarFeatures(neighborValues_mat, _neighborhoodWeights_eig);

    // if the cov matrix is not invertible but the intended distance matrix builds on that
    // we add small random noise to each dimension, uniformly sampled from (-abs(max(dim)), abs(max(dim))) * noiseMagnitude
    if (_distType == distance_metric::METRIC_BHATTACHARYYA && std::abs(meanCov_feat.cov_mat_det) < 1e-5f)
    {
        // define noise range per dimension
        float noiseMagnitude = 0.01f;  
        Eigen::VectorXf absMaxsDims = neighborValues_mat.cwiseAbs().rowwise().maxCoeff();
        Eigen::VectorXf noiseRangeDims = absMaxsDims * noiseMagnitude;
        for (auto& range : noiseRangeDims) { if (range < noiseMagnitude) range = noiseMagnitude; };
        
        // adding noise
        for (int d = 0; d < _numDims; d++) {
            neighborValues_mat.row(d) += randomVector(_neighborhoodSize, -1 * noiseRangeDims[d], noiseRangeDims[d]);;
        }

        meanCov_feat = compMultiVarFeatures(neighborValues_mat, _neighborhoodWeights_eig);

        //assert(std::abs(meanCov_feat.cov_mat_det) > 1e-5f); // this is often not the case
    }


    // save features
    MeanCov_feat feat = MeanCov_feat{ meanCov_feat.mean_vec, meanCov_feat.cov_mat, std::sqrt(meanCov_feat.cov_mat_det) };
    std::vector<IFeatureData*>* ib_featdata = _outFeatures.get_data_ptr();
    ib_featdata->at(pointInd) = new FeatureData<MeanCov_feat> (feat);

}


void FeatureExtraction::allNeighborhoodVals(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {

    // copy neighborValues into _outFeatures
    //_outFeatures.get_data_ptr()->at(pointInd) = new FeatureData<std::vector<float>>(neighborValues);
    _outFeatures.get_data_ptr()->at(pointInd) = new FeatureData<Eigen::MatrixXf>(Eigen::Map<Eigen::MatrixXf>(&neighborValues[0], _numDims, _neighborhoodSize));

}

void FeatureExtraction::allNeighborhoodIDs(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs) {
    // copy neighborIDs into _outFeatures
    _outFeatures.get_data_ptr()->at(pointInd) = new FeatureData<std::vector<int>>(neighborIDs);

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

    }

    // Normalize such that sum(_neighborhoodWeights) = 1
    NormVector(_neighborhoodWeights, std::accumulate(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 0.0f)); 
    _neighborhoodWeightsSum = std::accumulate(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 0.0f);

    _neighborhoodWeights_eig = Eigen::Map<Eigen::VectorXf>(_neighborhoodWeights.data(), _neighborhoodSize);
    assert(std::abs(_neighborhoodWeights_eig.sum() - _neighborhoodWeightsSum) < 0.01f);

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


Feature FeatureExtraction::output()
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
