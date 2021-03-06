#pragma once

#include "FeatureUtils.h"	// struct ImgSize;

#include <vector>

class SpidrParameters;
enum class loc_Neigh_Weighting : unsigned int;
enum class feature_type : unsigned int;

/*!
 * 
 * 
 */
class FeatureExtraction
{
public:
    FeatureExtraction();

    /*!
     * 
     * 
     * \return 
     */
    std::vector<float> output();

    void setNumLocNeighbors(size_t size);
    void setNeighborhoodWeighting(loc_Neigh_Weighting weighting);
    void setNumHistBins(size_t size);
    void stopFeatureCopmutation();
    bool requestedStop();

    loc_Neigh_Weighting getNeighborhoodWeighting();

    /*!
     * 
     * 
     * \param pointIds
     * \param attribute_data
     * \param params
     */
    void setup(const std::vector<unsigned int>& pointIds, const std::vector<float>& attribute_data, const SpidrParameters& params, std::vector<unsigned int>* backgroundIDsGlobal);

    /*!
	* Calculates features, basically calls initExtraction and extractFeatures
	*/
    void compute();

private:

    /*! Inits some summary values of the data depending on the feature type and resizes the output
     * The summary values are min, max, mean and var per dimension. Not all
     * summary values are computed for each feature type
    */
    void initExtraction();

    /*! Compute spatial features of the data
     * Depending on _featType, these can be classic texture features or other indicator of spatial association. 
     * Sets the output variables.
     */
    void extractFeatures();

    /*! Calculate Texture histograms
     * For each dimension compute a 1D histogram of the neighborhood values for pointID.
     * Sets _outFeatures.
     * 
     * \param pointInd
     * \param neighborValues
     */
    void calculateHistogram(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Calculate Local Indicator of Spatial Association features for each item
     * Compute Local Moran's I of the neighborhood values for pointID. 
     * Sets _outFeatures.
     * See doi:10.1111/j.1538-4632.1995.tb00338.x
     * \param pointInd
     * \param neighborValues
    */ 
    void calculateLISA(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Calculate Geary's C features for each item
     * Compute Geary's C of the neighborhood values for pointID.
     * Sets _outFeatures.
     * See doi:10.1111/j.1538-4632.1995.tb00338.x
     * \param pointInd
     * \param neighborValues
    */
    void calculateGearysC(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Compute the sum of dists^2 from one point to all others
     * See MVN-Reduce doi:10.2312/eurovisshort.20171126.x
     * \param pointInd
     * \param neighborValues
    */
    void calculateSumAllDist(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Sets the Feature per element to all it's neighbors attributes
     * The neighborhood is a square and centered around each item respectively
     * Padding is done by setting out-of-boundary values to 0
     * \param weighting
     */
    void allNeighborhoodVals(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Sets the Feature per element to all its neighbor's IDs
     * The neighborhood is a square and centered around each item respectively
     * Padding is done by setting out-of-boundary values to 0
     * \param weighting
     */
    void allNeighborhoodIDs(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Inits the neighborhood weighting
     * 
     * \param weighting
     */
    void weightNeighborhood(loc_Neigh_Weighting weighting);

    /*! Pointer to function that computer features
     * E.g. calculateHistogram or calculateLISA. 
     * It set in extractFeatures().
     */
    void(FeatureExtraction::*featFunct)(size_t, std::vector<float>, std::vector<int>);

    bool _stopFeatureComputation;                   /*!< Stops the computation (TODO: breaks the openmp parallel loop) */

    // Options 
    feature_type _featType;                         /*!< Type of feature to extract */
    size_t       _numFeatureValsPerPoint;           /*!< depending on the feature type, the features vector has a different length (scalar features vs vector features per dimension)> */
    size_t       _locNeighbors;                     /*!< Number of neighbors including center */
    size_t       _kernelWidth;                      /*!< Width of the kernel (2* _locNeighbors +1) */
    size_t       _neighborhoodSize;                 /*!< Square neighborhood centered around an item with _neighborhoodSize neighbors to the left, right, top and buttom */
    loc_Neigh_Weighting _neighborhoodWeighting;     /*!< Weighting type of neighborhood kernel */
    std::vector<float> _neighborhoodWeights;        /*!< Weightings of neighborhood kernel */
    float _neighborhoodWeightsSum;                  /*!< Sum of weightings in neighborhood kernel */
    size_t       _numHistBins;                      /*!< Number of bins in each histogram */
    bool _forceCalcBackgroundFeatures;              /*!< Force calculation of features for background data */

    // Data
    // Input
	ImgSize      _imgSize;                          /*!<> */
    size_t       _numDims;                          /*!<> */
    size_t       _numPoints;                        /*!<> */
    std::vector<unsigned int> _pointIds;            /*!<> */
    std::vector<float> _attribute_data;             /*!<> */
    std::vector<unsigned int>* _backgroundIDsGlobal;  /*!<> */
    std::vector<float> _minMaxVals;                 /*!< Extrema for each dimension/channel, i.e. [min_Ch0, max_Ch0, min_Ch1, max_Ch1, ...] */
    std::vector<float> _meanVals;                   /*!< Avg for each dimension/channel, i.e. [mean_Ch0, meam_Ch1, ...] */
    std::vector<float> _varVals;                    /*!< Variance estimate for each dimension/channel, i.e. [mean_Ch0, meam_Ch1, ...] */

    // Output
    /*! Features for each item.
    * In case of 1D histograms for each data point there are _inputData.getNumDimensions() histograms 
    * with _numHistBins values, i.e. size _numPoints * _numDims * _numHistBins.
    * Else, the features are the local Indicator of Spatial Association features for each item.
    */
    std::vector<float> _outFeatures;
};