#pragma once

#include <tuple>
#include <vector>
#include <string>   
#include "FeatureUtils.h"

class SpidrParameters;
enum class knn_library : size_t;
enum class distance_metric : size_t;
enum class feature_type : unsigned int;
enum class loc_Neigh_Weighting : unsigned int;

/*!
 * 
 * 
 */
class DistanceCalculation 
{
public:
    //DistanceCalculation();

    /*! Returns knn indices and distances as a tuple
     * Use with std::tie(_knn_indices, _knn_distances) = output()
     */
    std::tuple< std::vector<int>, std::vector<float>> output() const; // tuple of indices and dists
    
    std::vector<int> get_knn_indices() const;
    std::vector<float> get_knn_distances() const;

    void setKnnAlgorithm(knn_library knn);
    void setDistanceMetric(distance_metric metric);

    /*!
     * 
     * 
     * \param pointIds
     * \param dataFeatures
     * \param foregroundIDsGlobal
     * \param params
     */
    void setup(const Feature dataFeatures, const std::vector<unsigned int>& foregroundIDsGlobal, SpidrParameters& params);

    /*!
     * 
     * 
     */
    void compute();

private:

    void computekNN();

private:
    // Options
    feature_type _featureType;                      /*!<> */
    knn_library _knn_lib;                           /*!<> */
    distance_metric _knn_metric;                /*!<> */
    bool _normalize_data;                     /*!< if distance_metric is METRIC_COS the data must be normalized in order to use the HNSW InnerProduct Implementation> */
    size_t _nn;                               /*!<> */
    size_t _neighborhoodSize;                       /*!< might be used for some distance metrics */
    loc_Neigh_Weighting _neighborhoodWeighting;     /*!< used when calculating distance directly from high-dim points (_featureType is no feature/PCLOUD) */

    // Data
    // Input
    size_t _numDims;                                /*!<> */
    size_t _numPoints;                              /*!<> */
    size_t _numForegroundPoints;                              /*!<> */
    size_t _numHistBins;                            /*!<> */ // don't set this from the widget input. Instead you the value set in the feature extraction
    Feature _dataFeatures;
    size_t _numFeatureValsPerPoint;                 /*!< Feature Values per Point> */
    std::string _embeddingName;                     /*!< Name of the embedding */
    const float* _dataVecBegin;                     /*!< Points to the first element in the data vector> */
    std::vector<unsigned int> _foregroundIDsGlobal;  /*!<> */
    size_t _imgWidth;                                  /*!<> */

    // Output
    std::vector<int> _knn_indices;                      /*!<> */
    std::vector<float> _knn_distances;          /*!<> */
};