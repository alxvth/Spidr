#include "DistanceCalculation.h"

#include "SpidrAnalysisParameters.h"
#include "KNNUtils.h"
#include "EvalUtils.h"

#include "hnswlib/hnswlib.h"
#include "spdlog/spdlog-inl.h"

#include <chrono>
#include <algorithm>            // std::none_of
#include <iterator>             // std::make_move_iterator, find
#include <utility>              // std::move

DistanceCalculation::DistanceCalculation() :
    _knn_lib(knn_library::KNN_HNSW),
    _knn_metric(distance_metric::METRIC_QF)
{
}


void DistanceCalculation::setup(const Feature dataFeatures, const std::vector<unsigned int>& foregroundIDsGlobal, SpidrParameters& params) {
    spdlog::info("Distance calculation: Setup");
    _featureType = params._featureType;
    _numFeatureValsPerPoint = params._numFeatureValsPerPoint;

    // SpidrParameters
    _knn_lib = params._aknn_algorithm;
    _knn_metric = params._aknn_metric;
    _nn = params.get_nn();
    _neighborhoodSize = params._neighborhoodSize;    // square neighborhood with _numLocNeighbors to each side from the center
    _neighborhoodWeighting = params._neighWeighting;

    // Data
    // Input
    _numPoints = params._numPoints;
    _numForegroundPoints = params._numForegroundPoints;
    _numDims = params._numDims;
    _numHistBins = params._numHistBins;
    _embeddingName = params._embeddingName;
    _imgWidth = params._imgSize.width;

    _dataFeatures = dataFeatures;
    _foregroundIDsGlobal = foregroundIDsGlobal;

    // Output
    //_knn_indices.resize(_numForegroundPoints*_nn, -1);              // unnecessary, done in ComputeHNSWkNN
    //_knn_distances_squared.resize(_numForegroundPoints*_nn, -1);    // unnecessary, done in ComputeHNSWkNN

    assert(params.get_nn() == (size_t)(params.get_perplexity() * params.get_perplexity_multiplier() + 1));     // should be set in SpidrAnalysis::initializeAnalysisSettings

    spdlog::info("Distance calculation: Feature values per point: {0}, Number of NN to calculate {1}. Metric: {2}", _numFeatureValsPerPoint, _nn, static_cast<size_t> (_knn_metric));

    if (_numPoints != _numForegroundPoints)
        spdlog::info("Distance calculation: Do not consider {} background points", _numPoints - _numForegroundPoints);
}

void DistanceCalculation::compute() {
	spdlog::info("Distance calculation: Started");

    computekNN();

	spdlog::info("Distance calculation: Finished");

}

void DistanceCalculation::computekNN() {
    
	spdlog::info("Distance calculation: Setting up metric space");
    auto t_start_CreateHNSWSpace = std::chrono::steady_clock::now();

    // setup hsnw index
    hnswlib::SpaceInterface<float> *space = CreateHNSWSpace(_knn_metric, _numDims, _neighborhoodSize, _neighborhoodWeighting, _numFeatureValsPerPoint, _numHistBins, _imgWidth, _numForegroundPoints);
    assert(space != NULL);

    auto t_end_CreateHNSWSpace = std::chrono::steady_clock::now();
	spdlog::info("Distance calculation: Build time metric space (sec): {}", ((float)std::chrono::duration_cast<std::chrono::milliseconds> (t_end_CreateHNSWSpace - t_start_CreateHNSWSpace).count()) / 1000);
    spdlog::info("Distance calculation: Compute kNN");

    auto t_start_ComputeDist = std::chrono::steady_clock::now();

    if (_knn_lib == knn_library::KNN_HNSW) {
		spdlog::info("Distance calculation: HNSWLib for knn computation");

        std::tie(_knn_indices, _knn_distances_squared) = ComputeHNSWkNN(_dataFeatures, space, _numFeatureValsPerPoint, _foregroundIDsGlobal, _nn);

    }
    else if (_knn_lib == knn_library::KKN_EXACT) {
		spdlog::info("Distance calculation: Exact kNN computation");

        std::tie(_knn_indices, _knn_distances_squared) = ComputeExactKNN(_dataFeatures, space, _numFeatureValsPerPoint, _foregroundIDsGlobal, _nn);

    }
    else if (_knn_lib == knn_library::EVAL_KNN_EXACT) {
        // Save the entire distance matrix to disk. Then calc the exact kNN and perform the embedding
        // Note: You could also sort the distance matrix instead of recalculating it - but I'm lazy and will only use this for small data set where the performance is not an issue.

		spdlog::info("Distance calculation: Evaluation mode (exact) - Calc full distance matrix for writing to disk");
        std::vector<int> all_dists_indices_to_Disk;
        std::vector<float> all_distances_squared_to_Disk;
        std::tie(all_dists_indices_to_Disk, all_distances_squared_to_Disk) = ComputeFullDistMat(_dataFeatures, space, _numFeatureValsPerPoint, _foregroundIDsGlobal);

		spdlog::info("Distance calculation: Evaluation mode (exact) - Write full distance matrix to disk");

        // Write (full) distance matrix and IDs to disk
        std::string savePath = _embeddingName;
        std::string infoStr = "_nD_" + std::to_string(_numDims) + "_nP_" + std::to_string(_numForegroundPoints) + "_nN_" + std::to_string(_numForegroundPoints);
        writeVecToBinary(all_dists_indices_to_Disk, savePath + "_allInds" + infoStr + ".bin");
        writeVecToBinary(all_distances_squared_to_Disk, savePath + "_allDists" + infoStr + ".bin");

        // Write features to disk
        // TODO: This does not work anymore
        //infoStr = "_nFpP_" + std::to_string(_numFeatureValsPerPoint) + "_nP_" + std::to_string(_numForegroundPoints) + "_nD_" + std::to_string(_numDims);
        //writeVecToBinary(_dataFeatures.get_data_ptr(), savePath + "_features" + infoStr + ".bin");

		spdlog::info("Distance calculation: Evaluation mode (exact) - Calc exact knn distance matrix for embedding");
        std::tie(_knn_indices, _knn_distances_squared) = ComputeExactKNN(_dataFeatures, space, _numFeatureValsPerPoint, _foregroundIDsGlobal, _nn);

        // Write exact knn distances to disk
		spdlog::info("Distance calculation: Evaluation mode (exact) - Write knn distance matrix to disk");
        infoStr = "_nD_" + std::to_string(_numDims) + "_nP_" + std::to_string(_numForegroundPoints) + "_nN_" + std::to_string(_nn);
        writeVecToBinary(_knn_indices, savePath + "_knnInds" + infoStr + ".bin");
        writeVecToBinary(_knn_distances_squared, savePath + "_knnDists" + infoStr + ".bin");

    }
    else if (_knn_lib == knn_library::EVAL_KNN_HNSW) {
        // Save the akNN distance matrix to disk. 

		spdlog::info("Distance calculation: Evaluation mode (akNN) - HNSWLib for knn computation");
        std::tie(_knn_indices, _knn_distances_squared) = ComputeHNSWkNN(_dataFeatures, space, _numFeatureValsPerPoint, _foregroundIDsGlobal, _nn);

        // Write aknn distances to disk
		spdlog::info("Distance calculation: Evaluation mode (akNN) - Write aknn distance matrix to disk");
        std::string savePath = _embeddingName;
        std::string infoStr = "_nD_" + std::to_string(_numDims) + "_nP_" + std::to_string(_numForegroundPoints) + "_nN_" + std::to_string(_nn);
        writeVecToBinary(_knn_indices, savePath + "_aknnInds" + infoStr + ".bin");
        writeVecToBinary(_knn_distances_squared, savePath + "_aknnDists" + infoStr + ".bin");

    }
	else if (_knn_lib == knn_library::FULL_DIST_BRUTE_FORCE) {
		// Use entire distance matrix 
		spdlog::info("Distance calculation: Calc full distance matrix brute force");
		std::tie(_knn_indices, _knn_distances_squared) = ComputeFullDistMat(_dataFeatures, space, _numFeatureValsPerPoint, _foregroundIDsGlobal);


	}
	else
		throw std::runtime_error("Distance calculation: Unknown knn_library");

    auto t_end_ComputeDist = std::chrono::steady_clock::now();
	spdlog::info("Distance calculation: Computation duration (sec): {}", ((float)std::chrono::duration_cast<std::chrono::milliseconds> (t_end_ComputeDist - t_start_ComputeDist).count()) / 1000);

    // -1 would mark unset values
    assert(_knn_indices.size() == _numForegroundPoints * _nn);
    assert(_knn_distances_squared.size() == _numForegroundPoints * _nn);
    assert(std::none_of(_knn_indices.begin(), _knn_indices.end(), [](int i) {return i == -1; }));
    assert(std::none_of(_knn_distances_squared.begin(), _knn_distances_squared.end(), [](float i) {return i == -1.0f; }));

}

const std::tuple< std::vector<int>, std::vector<float>> DistanceCalculation::output() {
    return { _knn_indices, _knn_distances_squared };
}

std::vector<int> DistanceCalculation::get_knn_indices() {
    return _knn_indices;
}

std::vector<float> DistanceCalculation::get_knn_distances_squared() {
    return _knn_distances_squared;
}


void DistanceCalculation::setKnnAlgorithm(knn_library knn)
{
    _knn_lib = knn;
}

void DistanceCalculation::setDistanceMetric(distance_metric metric)
{
    _knn_metric = metric;
}