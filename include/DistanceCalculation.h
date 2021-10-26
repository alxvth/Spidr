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

    /*!
     * 
     * 
     */
    const std::tuple< std::vector<int>, std::vector<float>> output(); // tuple of indices and dists
    
    std::vector<int> get_knn_indices();
    std::vector<float> get_knn_distances_squared();

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
    std::vector<float> _knn_distances_squared;          /*!<> */
};