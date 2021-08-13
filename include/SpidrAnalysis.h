#pragma once

#include "TsneComputation.h"
#include "DistanceCalculation.h"
#include "FeatureExtraction.h"
#include "SpidrAnalysisParameters.h"
#include "FeatureUtils.h"
#include "KNNUtils.h"

#include <vector>
#include <string>
#include <tuple>     // std::tuple

/*!
 * 
 * 
 */
class SpidrAnalysis
{
public:
    SpidrAnalysis();

    /*!
     * Call me before calling initializeAnalysisSettings
     * 
     * \param attribute_data
     * \param pointIDsGlobal
     * \param numDimensions
     * \param imgSize
     * \param embeddingName
     * \param backgroundIDsGlobal ID of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation
     */
    void setupData(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
		const size_t numDimensions, const ImgSize imgSize, const std::string embeddingName, const std::vector<unsigned int>& backgroundIDsGlobal = std::vector<unsigned int>());

	/*! Set the parameters of the entire Analysis
	 * Use the input from e.g a GUI
	 *
	 * \param featType
	 * \param kernelInd
	 * \param numLocNeighbors Number of spatial Neighbors In Each Direction, thus 1 -> 3x3 neighborhood
	 * \param numHistBins
	 * \param aknnAlgInd
	 * \param aknnMetInd
	 * \param numIterations
	 * \param perplexity
	 * \param exaggeration
	 * \param expDecay
	 * \param forceCalcBackgroundFeatures
	 */
	void initializeAnalysisSettings(const feature_type featType, const loc_Neigh_Weighting kernelType, const size_t numLocNeighbors, const size_t numHistBins, \
		const knn_library aknnAlgType, const distance_metric aknnMetric, \
		const int numIterations, const int perplexity, const int exaggeration, const int expDecay, bool forceCalcBackgroundFeatures = false);


	/*! Compute feature extraction and embedding
	 * Calls computeFeatures, computekNN and computeEmbedding
	 */
	void compute();

	/*! Compute Features from raw data
	 * sets _dataFeats 
	 */
	void computeFeatures();

	/*! Based on _dataFeats, compute kNN
	 * sets _knn_indices and _knn_distances_squared
	 */
	void computekNN();

	/*! Compute t-SNE embedding
	 *
	 */
	void computeEmbedding();

    /*!
     * release openGL context of the t-SNE computation
     */
    void stopComputation();

    // Getter
    const size_t getNumForegroundPoints();
    const size_t getNumImagePoints();
    bool embeddingIsRunning();

    /*! Return embdding
	 *
     */
    const std::vector<float> &output();

	/*! Return embdding with background
	 * Checks if during setupData() any background points were specified and, if so, adds them into a corner in the embedding
	 */
	const std::vector<float> &outputWithBackground();

    const SpidrParameters getParameters();

	const Feature getDataFeatures();

    /* Returns _knn_indices, _knn_distances_squared, use with std::tie(_knnIds, _knnDists) = getKNN(); */
	const std::tuple<std::vector<int>, std::vector<float>> getKNN();

    /* Add bg points to emb, uses the ID info set for an instance of this class */
    void addBackgroundToEmbedding(std::vector<float>& emb, const std::vector<float>& emb_wo_bg);

private:
    

    // Setter

    /*! Sets feature type as in enum class feature_type in FeatureUtils.h
    *
    * \param feature_type_index, see enum class feature_type in FeatureUtils.h
    */
    void setFeatureType(const feature_type feature_type_index);

    /*! Sets feature type as in enum class loc_Neigh_Weighting in FeatureUtils.h
    *
    * \param loc_Neigh_Weighting_index, see enum class loc_Neigh_Weighting in FeatureUtils.h
    */
    void setKernelWeight(const loc_Neigh_Weighting loc_Neigh_Weighting_index);

    /*! Sets the number of spatially local pixel neighbors in each direction. Sets _params._kernelWidth and _params._neighborhoodSize as well*/
    void setNumLocNeighbors(const size_t num);

    /*! Sets the number of histogram bins */
    void setNumHistBins(const size_t num);

    /*! Sets knn algorithm type as in enum class feature_type in KNNUtils.h
    *
    * \param knn_library_index, see enum class feature_type in KNNUtils.h
    */
    void setKnnAlgorithm(const knn_library knn_library_index);

    /*! Sets knn algorithm type as in enum class distance_metric in KNNUtils.h
    *
    * \param distance_metric_index, see enum class distance_metric in KNNUtils.h
    */
    void setDistanceMetric(const distance_metric distance_metric_index);

    /*! Sets the perplexity and automatically determines the number of approximated kNN
    * nn = 3 * perplexity
    *
    * \param perplexity
    */
    void setPerplexity(const unsigned perplexity);
    /*! Sets the number of histogram bins */

    /*! Sets the number of gradient descent iteration */
    void setNumIterations(const unsigned numIt);

    /*! Sets the exageration during gradient descent */
    void setExaggeration(const unsigned exag);

    /*! Sets the exponential decay during gradient descent */
    void setExpDecay(const unsigned expDacay);

    /*! Sets the size of a feature, derived from other parameters */
    void setNumFeatureValsPerPoint(feature_type featType, size_t numDims, size_t numHistBins, size_t neighborhoodSize);

    void setForceCalcBackgroundFeatures(const bool forceCalcBackgroundFeatures);

private:
    // worker classes
    FeatureExtraction _featExtraction;					/*!<> */
    DistanceCalculation _distCalc;						/*!<> */
    TsneComputation _tsne;								/*!<> */
    
    // data and settings
    std::vector<float> _attribute_data;					/*!<> */
    std::vector<unsigned int> _pointIDsGlobal;			/*!<> */
    std::vector<unsigned int> _backgroundIDsGlobal;		/*!< ID of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation > */
    std::vector<unsigned int> _foregroundIDsGlobal;		/*!< ID of points which are used during the t-SNE embedding > */
    SpidrParameters _params;							/*!<> */
    std::vector<float> _emd_with_backgound;

	// features and knn
	Feature _dataFeats;						/*!<> */
	std::vector<int> _knn_indices ;						/*!<> */
	std::vector<float> _knn_distances_squared;			/*!<> */

};


