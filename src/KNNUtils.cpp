#pragma once
#include "KNNUtils.h"
#include "KNNDists.h"
#include "FeatureUtils.h"

#include "spdlog/spdlog-inl.h"
#include "hnswlib/hnswlib.h"

#include <cmath>     // std::sqrt, exp, floor
#include <numeric>   // std::inner_product, std:accumulate, std::iota
#include <algorithm> // std::find, fill, sort
#include <assert.h>


metricPair MakeMetricPair(feature_type ft, distance_metric dm) {
    return std::make_tuple(ft, dm);
}

distance_metric GetDistMetricFromMetricPair(const metricPair metricPair) {
    return std::get<1>(metricPair);
}

feature_type GetFeatureTypeFromMetricPair(const metricPair metricPair) {
    return std::get<0>(metricPair);
}

template<typename T>
T CalcMedian(std::vector<T> vec, size_t vecSize) {
	T median;

	size_t n = vecSize / 2;
	std::nth_element(vec.begin(), vec.begin() + n, vec.end());
	T vn = vec[n];
	if (vecSize % 2 == 1)	// uneven length
	{
		median = vn;
	}
	else					// even length, median is average of the central two items
	{
		std::nth_element(vec.begin(), vec.begin() + n - 1, vec.end());
		median = 0.5*(vn + vec[n - 1]);
	}

	return median;
}
template float CalcMedian<float>(std::vector<float> vec, size_t vecSize);


std::vector<float> BinSimilarities(size_t num_bins, bin_sim sim_type, float sim_weight) {
	std::vector<float> A(num_bins*num_bins, -1);
	size_t ground_dist_max = num_bins - 1;

	if (sim_type == bin_sim::SIM_EUC) {
		for (int i = 0; i < (int)num_bins; i++) {
			for (int j = 0; j < (int)num_bins; j++) {
				A[i * num_bins + j] = 1 - (float(std::abs(i - j)) / float(ground_dist_max));
			}
		}
	}
	else if (sim_type == bin_sim::SIM_EXP) {
		for (int i = 0; i < (int)num_bins; i++) {
			for (int j = 0; j < (int)num_bins; j++) {
				A[i * num_bins + j] = ::std::exp(-1 * sim_weight * float(std::abs(i - j)));
			}
		}
	}
	else if (sim_type == bin_sim::SIM_UNI) {
		std::fill(A.begin(), A.end(), 1);
	}

	// if there is a -1 in A, this value was not set (invalid ground_type option selected)
	assert(std::find(A.begin(), A.end(), -1) == A.end());

	return A;
}

template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN(const std::vector<T>& dataFeatures, Feature& dataFeaturesF, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, const std::vector<unsigned int>& foregroundIDsGlobal, unsigned int nn) {
    auto numForegroundPoints = foregroundIDsGlobal.size();

    std::vector<int> indices(numForegroundPoints * nn, -1);
    std::vector<float> distances_squared(numForegroundPoints * nn, -1);

	spdlog::info("Distance calculation: Build akNN Index");

    hnswlib::HierarchicalNSW<float> appr_alg(space, numForegroundPoints, 16, 200, 0);   // use default HNSW values for M, ef_construction random_seed

    // add data points: each data point holds indMultiplier values (number of feature values)
    // add the first data point outside the parallel loop
    //appr_alg.addPoint((void*)(dataFeatures.data() + foregroundIDsGlobal[0] * indMultiplier), (std::size_t) 0);
    appr_alg.addPoint((void*)(dataFeaturesF.at(foregroundIDsGlobal[0])), (std::size_t) 0);

#ifdef NDEBUG
    // This loop is for release mode, it's parallel loop implementation from hnswlib
    int num_threads = std::thread::hardware_concurrency();
    hnswlib::ParallelFor(1, numForegroundPoints, num_threads, [&](size_t i, size_t threadId) {
        //appr_alg.addPoint((void*)(dataFeatures.data() + (foregroundIDsGlobal[i] *indMultiplier)), (hnswlib::labeltype) i);
        appr_alg.addPoint((void*)(dataFeaturesF.at(foregroundIDsGlobal[i])), (hnswlib::labeltype) i);
    });
#else
// This loop is for debugging, when you want to sequentially add points
    for (int i = 1; i < numForegroundPoints; ++i)
    {
        //appr_alg.addPoint((void*)(dataFeatures.data() + (foregroundIDsGlobal[i] *indMultiplier)), (hnswlib::labeltype) i);
        appr_alg.addPoint((void*)(dataFeaturesF.at(foregroundIDsGlobal[i])), (hnswlib::labeltype) i);
    }
#endif
	spdlog::info("Distance calculation: Search akNN Index");

    // query dataset
#ifdef NDEBUG
#pragma omp parallel for
#endif
    for (int i = 0; i < numForegroundPoints; ++i)
    {
        // find nearest neighbors
        //auto top_candidates = appr_alg.searchKnn((void*)(dataFeatures.data() + (foregroundIDsGlobal[i] *indMultiplier)), (hnswlib::labeltype)nn);
        auto top_candidates = appr_alg.searchKnn((void*)(dataFeaturesF.at(foregroundIDsGlobal[i])), (hnswlib::labeltype)nn);
        while (top_candidates.size() > nn) {
            top_candidates.pop();
        }

        assert(top_candidates.size() == nn);

        // save nn in _knn_indices and _knn_distances_squared 
        auto *distances_offset = distances_squared.data() + (i*nn);
        auto indices_offset = indices.data() + (i*nn);
        int j = 0;
        while (top_candidates.size() > 0) {
            auto rez = top_candidates.top();
            distances_offset[nn - j - 1] = rez.first;
            indices_offset[nn - j - 1] = appr_alg.getExternalLabel(rez.second);
            top_candidates.pop();
            ++j;
        }
    }

    return std::make_tuple(indices, distances_squared);
}
// Resolve linker errors with explicit instantiation, https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
template std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN<float>(const std::vector<float>& dataFeatures, Feature& dataFeaturesF, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, const std::vector<unsigned int>& foregroundIDsGlobal, unsigned int nn);
template std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN<unsigned int>(const std::vector<unsigned int>& dataFeatures, Feature& dataFeaturesF, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, const std::vector<unsigned int>& foregroundIDsGlobal, unsigned int nn);


template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeExactKNN(const std::vector<T> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, const std::vector<unsigned int>& foregroundIDsGlobal, unsigned int nn, bool fullDistMat) {
    auto numForegroundPoints = foregroundIDsGlobal.size();
    
    std::vector<std::pair<int, float>> indices_distances(numForegroundPoints);
	std::vector<int> knn_indices(numForegroundPoints*nn, -1);
	std::vector<float> knn_distances_squared(numForegroundPoints*nn, -1.0f);

	hnswlib::DISTFUNC<float> distfunc = space->get_dist_func();
	void* params = space->get_dist_func_param();

	// only used if fullDistMat == true 
	std::vector<int> idx_row(nn);
	std::iota(idx_row.begin(), idx_row.end(), 0);

	// For each point, calc distances to all other
	// and take the nn smallest as kNN
	for (int i = 0; i < (int)numForegroundPoints; i++) {
		// Calculate distance to all points  using the respective metric
#ifdef NDEBUG
#pragma omp parallel for
#endif
		for (int j = 0; j < (int)numForegroundPoints; j++) {
			indices_distances[j] = std::make_pair(j, distfunc(dataFeatures.data() + foregroundIDsGlobal[i] * featureSize, dataFeatures.data() + foregroundIDsGlobal[j] * featureSize, params));
		}

		if (!fullDistMat)
		{
			// compute knn, not full distance matrix
			assert(nn < numForegroundPoints);
			// sort all distances to point i
			std::sort(indices_distances.begin(), indices_distances.end(), [](std::pair<int, float> a, std::pair<int, float> b) {return a.second < b.second; });
	
			// Take the first nn indices 
			std::transform(indices_distances.begin(), indices_distances.begin() + nn, knn_indices.begin() + i * nn, [](const std::pair<int, float>& p) { return p.first; });
		}
		else
		{
			assert(nn == numForegroundPoints);
			// for full distance matrix, sort the indices depending on the distances
			std::sort(idx_row.begin(), idx_row.end(), [&indices_distances](int i1, int i2) {return indices_distances[i1].second < indices_distances[i2].second; });

			// Take the first nn indices (just copy them from the sorted indices)
			std::copy(idx_row.begin(), idx_row.begin() + nn, knn_indices.begin() + i * nn);
		}

		// Take the first nn distances 
		std::transform(indices_distances.begin(), indices_distances.begin() + nn, knn_distances_squared.begin() + i * nn, [](const std::pair<int, float>& p) { return p.second; });
	}

	return std::make_tuple(knn_indices, knn_distances_squared);
}
// Resolve linker errors with explicit instantiation, https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
template std::tuple<std::vector<int>, std::vector<float>> ComputeExactKNN<float>(const std::vector<float> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, const std::vector<unsigned int>& foregroundIDsGlobal, unsigned int nn, bool sort);
template std::tuple<std::vector<int>, std::vector<float>> ComputeExactKNN<unsigned int>(const std::vector<unsigned int> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, const std::vector<unsigned int>& foregroundIDsGlobal, unsigned int nn, bool sort);

template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeFullDistMat(const std::vector<T> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, const std::vector<unsigned int>& foregroundIDsGlobal) {
	// set nn = numForegroundPoints and don't sort the nn
	return ComputeExactKNN(dataFeatures, space, featureSize, foregroundIDsGlobal, foregroundIDsGlobal.size(), true);
}
// Resolve linker errors with explicit instantiation, https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
template std::tuple<std::vector<int>, std::vector<float>> ComputeFullDistMat<float>(const std::vector<float> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, const std::vector<unsigned int>& foregroundIDsGlobal);
template std::tuple<std::vector<int>, std::vector<float>> ComputeFullDistMat<unsigned int>(const std::vector<unsigned int> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, const std::vector<unsigned int>& foregroundIDsGlobal);


hnswlib::SpaceInterface<float>* CreateHNSWSpace(const distance_metric knn_metric, const size_t numDims, const size_t neighborhoodSize, const loc_Neigh_Weighting neighborhoodWeighting, const size_t featureValsPerPoint, const size_t numHistBins, const float* dataVecBegin, int imgWidth, int numPoints) {
    // chose distance metric
    hnswlib::SpaceInterface<float> *space = NULL;
    if (knn_metric == distance_metric::METRIC_QF)
    {
        assert(numHistBins > 0);
		spdlog::info("Distance calculation: QFSpace as vector feature");
        space = new hnswlib::QFSpace(numDims, numHistBins, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_HEL)
    {
        assert(numHistBins > 0);
		spdlog::info("Distance calculation: HellingerSpace as vector feature metric");
        space = new hnswlib::HellingerSpace(numDims, numHistBins, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_EUC)
    {
		spdlog::info("Distance calculation: EuclidenSpace (L2Space) as scalar feature metric");
        space = new hnswlib::L2FeatSpace(numDims);  // featureValsPerPoint = numDims
    }
    else if (knn_metric == distance_metric::METRIC_CHA)
    {
        assert(dataVecBegin != NULL);
		spdlog::info("Distance calculation: EuclidenSpace (ChamferSpace, Chamfer distance)");
        space = new hnswlib::ChamferSpace(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_SSD)
    {
        assert(dataVecBegin != NULL);
		spdlog::info("Distance calculation: EuclidenSpace (Sum of Squared Distances)");
        space = new hnswlib::SSDSpace(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_HAU)
    {
        assert(dataVecBegin != NULL);
		spdlog::info("Distance calculation: EuclidenSpace (Hausdorff)");
        space = new hnswlib::HausdorffSpace(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_HAU_med)
    {
        assert(dataVecBegin != NULL);
		spdlog::info("Distance calculation: EuclidenSpace (Hausdorff, med)");
        space = new hnswlib::HausdorffSpace_median(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_BHATTACHARYYA)
    {
        assert(dataVecBegin != NULL);
        spdlog::info("Distance calculation: BhattacharyyaSpace (Distance between means and covariance matrices)");
        space = new hnswlib::Bhattacharyya_Space();
    }
    else
		spdlog::error("Distance calculation: ERROR: Distance metric unknown.");

    return space;
}

