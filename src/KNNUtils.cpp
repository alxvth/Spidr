#pragma once
#include "KNNUtils.h"


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
std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN(const std::vector<T>& dataFeatures, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, size_t numPoints, unsigned int nn) {

    std::vector<int> indices(numPoints * nn, -1);
    std::vector<float> distances_squared(numPoints * nn, -1);

    //qDebug() << "ComputeHNSWkNN: Build akNN Index";

    hnswlib::HierarchicalNSW<float> appr_alg(space, numPoints);   // use default HNSW values for M, ef_construction random_seed

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
    //qDebug() << "ComputeHNSWkNN: Search akNN Index";

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


hnswlib::SpaceInterface<float>* CreateHNSWSpace(const distance_metric knn_metric, const size_t numDims, const size_t neighborhoodSize, const loc_Neigh_Weighting neighborhoodWeighting, const size_t featureValsPerPoint, const size_t numHistBins, const float* dataVecBegin, float weight, int imgWidth, int numPoints) {
    // chose distance metric
    hnswlib::SpaceInterface<float> *space = NULL;
    if (knn_metric == distance_metric::METRIC_QF)
    {
        assert(numHistBins > 0);
        //qDebug() << "Distance calculation: QFSpace as vector feature";
        space = new hnswlib::QFSpace(numDims, numHistBins, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_EMD)
    {
        assert(numHistBins > 0);
        //qDebug() << "Distance calculation: EMDSpace as vector feature";
        space = new hnswlib::EMDSpace(numDims, numHistBins, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_HEL)
    {
        assert(numHistBins > 0);
        //qDebug() << "Distance calculation: HellingerSpace as vector feature metric";
        space = new hnswlib::HellingerSpace(numDims, numHistBins, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_EUC)
    {
        //qDebug() << "Distance calculation: EuclidenSpace (L2Space) as scalar feature metric";
        space = new hnswlib::L2Space(numDims);  // featureValsPerPoint = numDims
    }
    else if (knn_metric == distance_metric::METRIC_CHA)
    {
        assert(dataVecBegin != NULL);
        //qDebug() << "Distance calculation: EuclidenSpace (ChamferSpace, Chamfer distance)";
        space = new hnswlib::ChamferSpace(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_SSD)
    {
        assert(dataVecBegin != NULL);
        //qDebug() << "Distance calculation: EuclidenSpace (Sum of Squared Distances)";
        space = new hnswlib::SSDSpace(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_HAU)
    {
        assert(dataVecBegin != NULL);
        //qDebug() << "Distance calculation: EuclidenSpace (Hausdorff)";
        space = new hnswlib::HausdorffSpace(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_MVN)
    {
        assert(dataVecBegin != NULL);
        //qDebug() << "Distance calculation: MVN-Reduce - Spatial and Attribute distancec combined with weight " << weight;
        space = new hnswlib::MVNSpace(numDims, weight, imgWidth, dataVecBegin, numPoints);
    }
    else if (knn_metric == distance_metric::METRIC_HAU_min)
    {
        assert(dataVecBegin != NULL);
        //qDebug() << "Distance calculation: EuclidenSpace (Hausdorff, min)";
        space = new hnswlib::HausdorffSpace_min(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_HAU_med)
    {
        assert(dataVecBegin != NULL);
        //qDebug() << "Distance calculation: EuclidenSpace (Hausdorff, med)";
        space = new hnswlib::HausdorffSpace_median(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_HAU_medmed)
    {
        assert(dataVecBegin != NULL);
        //qDebug() << "Distance calculation: EuclidenSpace (Hausdorff, med)";
        space = new hnswlib::HausdorffSpace_medianmedian(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else if (knn_metric == distance_metric::METRIC_HAU_minmax)
    {
        assert(dataVecBegin != NULL);
        //qDebug() << "Distance calculation: EuclidenSpace (Hausdorff, minmax)";
        space = new hnswlib::HausdorffSpace_minmax(numDims, neighborhoodSize, neighborhoodWeighting, dataVecBegin, featureValsPerPoint);
    }
    else
        //qDebug() << "Distance calculation: ERROR: Distance metric unknown.";

    return space;
}

const size_t NumFeatureValsPerPoint(const feature_type featureType, const size_t numDims, const size_t numHistBins, const size_t neighborhoodSize) {
    size_t featureSize = 0;
    switch (featureType) {
    case feature_type::TEXTURE_HIST_1D: featureSize = numDims * numHistBins; break;
    case feature_type::LOCALMORANSI:            // same as Geary's C
    case feature_type::LOCALGEARYC:          featureSize = numDims; break;
    case feature_type::PCLOUD:          featureSize = neighborhoodSize; break; // numDims * neighborhoodSize for copying data instead of IDs
    case feature_type::MVN:             featureSize = numDims + 2; break; // for each point: attr vals and x&y pos
    }

    return featureSize;
}
