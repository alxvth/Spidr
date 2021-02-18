#pragma once

#include <omp.h>

#include <cmath>     // std::sqrt, exp, floor
#include <numeric>   // std::inner_product, std:accumulate 
#include <algorithm> // std::find, fill, sort
#include <utility>   // std::pair
#include <vector>
#include <tuple>     // std::tuple, std::get, std::tie, std::ignore
#include <assert.h>

#include "SpidrAnalysisParameters.h"

typedef std::tuple<feature_type, distance_metric> metricPair;
namespace hnswlib {
	template<typename MTYPE> class SpaceInterface;
}

/*!
 * Used in SpidrSettingsWidget to set the distanceMetric/Feature Type pair
 * Casts enum classes to unsigned int
 * \param ft feature_type
 * \param dm distance_metric
 * \return metricPair -> std::tuple<feature_type, distance_metric>
 */
metricPair MakeMetricPair(feature_type ft, distance_metric dm);

distance_metric GetDistMetricFromMetricPair(const metricPair metricPair);
feature_type GetFeatureTypeFromMetricPair(const metricPair metricPair);

/*!
 * Used in SpidrSettingsWidget to set the distanceMetric/Feature Type pair
 * Casts enum classes to unsigned int
 * \param vec
 * \param vecSize
 * \return median
 */
template<typename T>
T CalcMedian(std::vector<T> vec, size_t vecSize);
template<typename T>
T CalcMedian(std::vector<T> vec) { CalcMedian(vec, vec.size()); }

/*!
 * Computes the similarities of bins of a 1D histogram.
 *
 * Entry A_ij refers to the sim between bin i and bin j. The diag entries should be 1, all others <= 1.
 *
 * \param num_bins    
 * \param sim_type type of ground distance calculation
 * \param sim_weight Only comes into play for ground_type = SIM_EXP, might be set to (0.5 * sd of all data * ground_dist_max^2) as im doi:10.1006/cviu.2001.0934
 * \return Matrix of neighborhood_width*neighborhood_width (stored in a vector) 
 */
std::vector<float> BinSimilarities(size_t num_bins, bin_sim sim_type = bin_sim::SIM_EUC, float sim_weight = 1);

/*! Compute approximated kNN with a custom metric using HNSWLib
 * \param dataFeatures Features used for distance calculation, dataFeatures->size() == (numPoints * indMultiplier)
 * \param space HNSWLib metric space
 * \param featureSize Size of one data item features
 * \param numPoints Number of points in the data
 * \param nn Number of kNN to compute
 * \return Tuple of knn Indices and respective squared distances
*/
template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN(const std::vector<T>& dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints, unsigned int nn);

/*! Compute exact kNNs 
 * Calculate the distances between all point pairs and find closest neighbors
 * \param dataFeatures Features used for distance calculation, dataFeatures->size() == (numPoints * indMultiplier)
 * \param space HNSWLib metric space
 * \param featureSize Size of one data item features
 * \param numPoints Number of points in the data
 * \param nn Number of nearest neighbors
 * \param sort Whether to sort the nearest neighbor distances. Default is true. Set to false if nn == numPoints and you want to calculate the full distance matrix
 * \return Tuple of indices and respective squared distances
*/
template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeExactKNN(const std::vector<T> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints, unsigned int nn, bool sort = true);

/*! Compute the full distance matrix between all data points
 * Calls ComputeExactKNN with the correct parameters, basically syntactic sugar
 * \param dataFeatures Features used for distance calculation, dataFeatures->size() == (numPoints * indMultiplier)
 * \param space HNSWLib metric space
 * \param featureSize Size of one data item features
 * \param numPoints Number of points in the data
 * \return Tuple of indices and respective squared distances
*/
template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeFullDistMat(const std::vector<T> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints);

/*! Creates a metric space used by HNSWLib to build a kNN index
 * 
 * \param knn_metric distance metric to compare two points with
 * \param numDims Number of data channels
 * \param neighborhoodSize Size of neighborhood, must be a perfect square
 * \param neighborhoodWeighting Featureless distances use the weighting
 * \param featureValsPerPoint used for data_size_
 * \param numHistBins Number of histogram bins of feature type is a vector i.e. histogram
 * \param dataVecBegin Used for PC distance where features are just IDs of the actual data
 * \param weight 
 * \param imgWidth 
 * \param numPoints 
 * \return A HNSWLib compatible SpaceInterface, which is used as the basis to compare two points
 */
hnswlib::SpaceInterface<float>* CreateHNSWSpace(const distance_metric knn_metric, const size_t numDims, const size_t neighborhoodSize, const loc_Neigh_Weighting neighborhoodWeighting, const size_t featureValsPerPoint, const size_t numHistBins=0, const float* dataVecBegin = NULL, float weight = 0, int imgWidth = 0, int numPoints = 0);

