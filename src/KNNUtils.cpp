#pragma once
#include "KNNUtils.h"
#include "KNNDists.h"
#include "spdlog/spdlog-inl.h"
#include "hnswlib/hnswlib.h"


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
std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN(const std::vector<T>& dataFeatures, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, size_t numPoints, unsigned int nn) {

    std::vector<int> indices(numPoints * nn, -1);
    std::vector<float> distances_squared(numPoints * nn, -1);

	spdlog::info("Distance calculation: Build akNN Index");

    hnswlib::HierarchicalNSW<float> appr_alg(space, numPoints, 16, 200, 0);   // use default HNSW values for M, ef_construction random_seed

    // add data points: each data point holds _numDims*_numHistBins values
    appr_alg.addPoint((void*)dataFeatures.data(), (std::size_t) 0);


#ifdef NDEBUG
    // This loop is for release mode, it's parallel loop implementation from hnswlib
    int num_threads = std::thread::hardware_concurrency();
    hnswlib::ParallelFor(1, numPoints, num_threads, [&](size_t i, size_t threadId) {
        appr_alg.addPoint((void*)(dataFeatures.data() + (i*indMultiplier)), (hnswlib::labeltype) i);
    });
#else
// This loop is for debugging, when you want to sequentially add points
    for (int i = 1; i < numPoints; ++i)
    {
        appr_alg.addPoint((void*)(dataFeatures.data() + (i*indMultiplier)), (hnswlib::labeltype) i);
    }
#endif
	spdlog::info("Distance calculation: Search akNN Index");

    // query dataset
#ifdef NDEBUG
#pragma omp parallel for
#endif
    for (int i = 0; i < numPoints; ++i)
    {
        // find nearest neighbors
        auto top_candidates = appr_alg.searchKnn((void*)(dataFeatures.data() + (i*indMultiplier)), (hnswlib::labeltype)nn);
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
template std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN<float>(const std::vector<float>& dataFeatures, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, size_t numPoints, unsigned int nn);
template std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN<unsigned int>(const std::vector<unsigned int>& dataFeatures, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, size_t numPoints, unsigned int nn);


template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeExactKNN(const std::vector<T> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints, unsigned int nn, bool sort) {
	std::vector<std::pair<int, float>> indices_distances;
	std::vector<int> knn_indices;
	std::vector<float> knn_distances_squared;

	indices_distances.resize(numPoints);
	knn_indices.resize(numPoints*nn, -1);
	knn_distances_squared.resize(numPoints*nn, -1.0f);

	hnswlib::DISTFUNC<float> distfunc = space->get_dist_func();
	void* params = space->get_dist_func_param();

	// For each point, calc distances to all other
	// and take the nn smallest as kNN
	for (int i = 0; i < (int)numPoints; i++) {
		// Calculate distance to all points  using the respective metric
#ifdef NDEBUG
#pragma omp parallel for
#endif
		for (int j = 0; j < (int)numPoints; j++) {
			indices_distances[j] = std::make_pair(j, distfunc(dataFeatures.data() + i * featureSize, dataFeatures.data() + j * featureSize, params));
		}

		if (sort)
		{
			// sort all distances to point i
			std::sort(indices_distances.begin(), indices_distances.end(), [](std::pair<int, float> a, std::pair<int, float> b) {return a.second < b.second; });
		}

		// Take the first nn indices 
		std::transform(indices_distances.begin(), indices_distances.begin() + nn, knn_indices.begin() + i * nn, [](const std::pair<int, float>& p) { return p.first; });
		// Take the first nn distances 
		std::transform(indices_distances.begin(), indices_distances.begin() + nn, knn_distances_squared.begin() + i * nn, [](const std::pair<int, float>& p) { return p.second; });
	}

	return std::make_tuple(knn_indices, knn_distances_squared);
}
// Resolve linker errors with explicit instantiation, https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
template std::tuple<std::vector<int>, std::vector<float>> ComputeExactKNN<float>(const std::vector<float> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, size_t numPoints, unsigned int nn, bool sort);
template std::tuple<std::vector<int>, std::vector<float>> ComputeExactKNN<unsigned int>(const std::vector<unsigned int> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints, unsigned int nn, bool sort);

template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeFullDistMat(const std::vector<T> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints) {
	// set nn = numPoints and sort = false
	return ComputeExactKNN(dataFeatures, space, featureSize, numPoints, numPoints, false);
}
// Resolve linker errors with explicit instantiation, https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
template std::tuple<std::vector<int>, std::vector<float>> ComputeFullDistMat<float>(const std::vector<float> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints);
template std::tuple<std::vector<int>, std::vector<float>> ComputeFullDistMat<unsigned int>(const std::vector<unsigned int> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints);


hnswlib::SpaceInterface<float>* CreateHNSWSpace(const distance_metric knn_metric, const size_t numDims, const size_t neighborhoodSize, const loc_Neigh_Weighting neighborhoodWeighting, const size_t featureValsPerPoint, const size_t numHistBins, const float* dataVecBegin, float weight, int imgWidth, int numPoints) {
    // chose distance metric
    hnswlib::SpaceInterface<float> *space = NULL;
    if (knn_metric == distance_metric::METRIC_QF)
    {
        assert(numHistBins > 0);
		spdlog::info("Distance calculation: QFSpace as vector feature");
        space = new hnswlib::QFSpace(numDims, numHistBins, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_EMD)
    {
        assert(numHistBins > 0);
		spdlog::info("Distance calculation: EMDSpace as vector feature");
        space = new hnswlib::EMDSpace(numDims, numHistBins, featureValsPerPoint);
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
        space = new hnswlib::L2Space(numDims);  // featureValsPerPoint = numDims
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
    else if (knn_metric == distance_metric::METRIC_MVN)
    {
        assert(dataVecBegin != NULL);
		spdlog::info("Distance calculation: MVN-Reduce - Spatial and Attribute distancec combined with weight {}", weight);
        space = new hnswlib::MVNSpace(numDims, weight, imgWidth, dataVecBegin, numPoints);
    }
    else if (knn_metric == distance_metric::METRIC_HAU_min)
    {
        assert(dataVecBegin != NULL);
		spdlog::info("Distance calculation: EuclidenSpace (Hausdorff, min)");
        space = new hnswlib::HausdorffSpace_min(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_HAU_med)
    {
        assert(dataVecBegin != NULL);
		spdlog::info("Distance calculation: EuclidenSpace (Hausdorff, med)");
        space = new hnswlib::HausdorffSpace_median(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_HAU_medmed)
    {
        assert(dataVecBegin != NULL);
		spdlog::info("Distance calculation: EuclidenSpace (Hausdorff, med)");
        space = new hnswlib::HausdorffSpace_medianmedian(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_HAU_minmax)
    {
        assert(dataVecBegin != NULL);
		spdlog::info("Distance calculation: EuclidenSpace (Hausdorff, minmax)");
        space = new hnswlib::HausdorffSpace_minmax(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else
		spdlog::error("Distance calculation: ERROR: Distance metric unknown.");

    return space;
}

