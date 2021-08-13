#pragma once

#include <string>
#include <stdexcept>

#include "spdlog/spdlog-inl.h"

typedef struct ImgSize {
	int width;
	int height;

	ImgSize() : width(-1), height(-1) {};
	ImgSize(int width, int height) : width(width), height(height) {};

} ImgSize;


/*! kNN library that is used kNN computations
 * The librarires are extended in order to work with different feature types
 */
enum class knn_library : size_t
{
	EVAL_KNN_EXACT,     /*!< No knn library in use, saves full dist matrix and IDs as well as features to disk */
	EVAL_KNN_HNSW,      /*!< HNSWLib in use, saves akNN distance matrix and IDs */
	KKN_EXACT,          /*!< No knn library in use, no approximation i.e. exact kNN computation */
	KNN_HNSW,			/*!< HNSWLib */
	FULL_DIST_BRUTE_FORCE,	/*!< straight forward brute force computation of the entire distance matric - not intended to be used with t-SNE but rather if you are interested in the distance matrix and not the knn */
};

/*! Defines the distance metric
* add respective entries to logging::distance_metric_name
 */
enum class distance_metric : size_t
{
	METRIC_QF,       /*!< Quadratic form distance */
	METRIC_HEL,      /*!< Hellinger distance */
	METRIC_EUC,      /*!< Euclidean distance */
	METRIC_CHA,      /*!< Chamfer distance (point cloud)*/
	METRIC_SSD,      /*!< Sum of squared distances (point cloud)*/
	METRIC_HAU,      /*!< Hausdorff distance (point cloud)*/
	METRIC_HAU_med,      /*!< Hausdorff distance (point cloud) but with median instead of max*/
	METRIC_BHATTACHARYYA,      /*!< Bhattacharyya distance between two multivariate normal distributions, https://en.wikipedia.org/wiki/Bhattacharyya_distance */
	METRIC_DETMATRATIO,      /*!< Deteterminant Ratio part of Bhattacharyya distance, i.e. Bat distance between two distributions with the same mean */
	METRIC_CMD_covmat,      /*!< Correlation Matrix distance http://dx.doi.org/10.1109/VETECS.2005.1543265 */
	METRIC_FRECHET_Gen,      /*!< The Fréchet distance between multivariate normal distributions, https://doi.org/10.1016/0047-259X(82)90077-X */
    METRIC_FRECHET_CovMat,      /*!< The Fréchet distance between multivariate normal distributions but ignoring the means, https://doi.org/10.1016/0047-259X(82)90077-X */
	METRIC_FROBENIUS_CovMat,      /*!< Frobenius norm of element-wise differences between covmatrices */

};

namespace logging {
    std::string distance_metric_name(const distance_metric& metric);
}

/*!
 * Types of ground distance calculation that are used as the basis for bin similarities
 */
enum class bin_sim : size_t
{
	SIM_EUC,    /*!< 1 - sqrt(Euclidean distance between bins)/(Max dist) */
	SIM_EXP,    /*!< exp(-(Euclidean distance between bins)^2/(Max dist)) */
	SIM_UNI,    /*!< 1 (uniform) */
};


/*! Types of neighborhood features
 *
 * Adding a new feature type? Make sure to adjust NumFeatureValsPerPoint()
 */
enum class feature_type : unsigned int
{
	TEXTURE_HIST_1D = 0,    /*!< Histograms of data point neighborhood, vector feature */
	LOCALMORANSI = 1,       /*!< Local Moran's I (Local Indicator of Spatial Associations), scalar feaure */
	LOCALGEARYC = 2,        /*!< Local Geary's C (Local Indicator of Spatial Associations), scalar feature */
	PCLOUD = 3,             /*!< Point cloud, i.e. just the neighborhood, no transformations*/
	MULTIVAR_NORM = 4,      /*!< Mean and covariance matrix  */
};

// Heuristic for setting the histogram bin size
enum class histBinSizeHeuristic : unsigned int
{
	MANUAL = 0,    /*!< Manually  adjust histogram bin size */
	SQRT = 1,      /*!< ceil(sqrt(n)), n = neighborhood size */
	STURGES = 2,   /*!< ceil(log_2(n))+1, n = neighborhood size */
	RICE = 3,      /*!< ceil(2*pow(n, 1/3)), n = neighborhood size */
};


/*! Weighting of local neighborhoods
 * Used e.g. in histogram creation, spatial weighting in LOCALMORANSI and Point cloud distance
 */
enum class loc_Neigh_Weighting : unsigned int
{
	WEIGHT_UNIF = 0,    /*!< Uniform weighting (all 1) */
	WEIGHT_BINO = 1,    /*!< Weighting binomial approximation of 2D gaussian */
	WEIGHT_GAUS = 2,    /*!< Weighting given by 2D gaussian */
};

/*!
 *
 *
 */
enum class norm_vec : unsigned int
{
	NORM_NONE = 0,   /*!< No normalization */
	NORM_MAX = 1,   /*!< Normalization such that max = 1 (usually center value) */
	NORM_SUM = 2,   /*!< Normalization such that sum = 1 */
};

/*! Heuristics for determining the Number of histogram bins based on the number of data points
* https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
 */
enum class bin_size : unsigned int
{
	MANUAL = 0,     /*!< define manually> */
	SQRT = 1,       /*!< sqrt(n) > */
	STURGES = 2,    /*!< ceil(log2(n)) + 1> */
	RICE = 3,       /*!< 2 * sqrt3(n) > */
};


/*! Calculates the size of an feature wrt to the feature type
 * Used as a step size for adding points to an HNSWlib index
 *
 * \param featureType type of feature (e.g. scalar LOCALMORANSI or vector Texture Histogram)
 * \param numDims Number of data channels
 * \param numHistBins Number of histogram bins of feature type is a vector i.e. histogram
 * \param neighborhoodSize Size of neighborhood, must be a perfect square
 * \return
 */
static const size_t NumFeatureValsPerPoint(const feature_type featureType, const size_t numDims, const size_t numHistBins, const size_t neighborhoodSize) {
	size_t featureSize = 0;
	switch (featureType) {
	case feature_type::TEXTURE_HIST_1D: featureSize = numDims * numHistBins; break;
	case feature_type::LOCALMORANSI:    // same as Geary's C
	case feature_type::LOCALGEARYC:     featureSize = numDims; break;
	case feature_type::PCLOUD:          featureSize = neighborhoodSize; break; // numDims * neighborhoodSize for copying data instead of IDs
	case feature_type::MULTIVAR_NORM:   featureSize = numDims + numDims* numDims + 2; break; // channel-wise means + covaraince matrix
    default: throw std::runtime_error("No feature size defined for this feature");
	}

	return featureSize;
}



/*!
 * Stores all parameters used in the Spatial Analysis.
 *
 * Used to set parameters for FeatureExtraction, DistanceCalculation and TsneComputatio
 */
class SpidrParameters {
public:
    SpidrParameters() :
        _nn(-1), _numPoints(-1), _numDims(-1), _imgSize(-1, -1), _embeddingName(""),
        _featureType(feature_type::TEXTURE_HIST_1D), _neighWeighting(loc_Neigh_Weighting::WEIGHT_UNIF), _numLocNeighbors(-1), _numHistBins(-1),
        _kernelWidth(0), _neighborhoodSize(0), _numFeatureValsPerPoint(0), _forceCalcBackgroundFeatures(false),
		_aknn_algorithm(knn_library::KNN_HNSW), _aknn_metric(distance_metric::METRIC_QF),
		_perplexity(30), _perplexity_multiplier(3), _numIterations(1000), _exaggeration(250)
	{}

	SpidrParameters(size_t numPoints, size_t numDims, ImgSize imgSize, std::string embeddingName, const float* dataVecBegin,
		feature_type featureType, loc_Neigh_Weighting neighWeighting, size_t numLocNeighbors, size_t numHistBins,
		knn_library aknn_algorithm, distance_metric aknn_metric, float MVNweight,
		float perplexity, int numIterations, int exaggeration, bool forceCalcBackgroundFeatures = false) :
		_numPoints(numPoints), _numDims(numDims), _imgSize(imgSize), _embeddingName(embeddingName),
		_featureType(featureType), _neighWeighting(neighWeighting), _numLocNeighbors(numLocNeighbors), _numHistBins(numHistBins),
		_aknn_algorithm(aknn_algorithm), _aknn_metric(aknn_metric), _forceCalcBackgroundFeatures(forceCalcBackgroundFeatures),
		_perplexity(perplexity), _perplexity_multiplier(3), _numIterations(numIterations), _exaggeration(exaggeration)
	{
		update_nn();	// sets nn based on perplexity
		_numForegroundPoints = numPoints; // No background default to all points in the foreground
		_kernelWidth = (2 * _numLocNeighbors) + 1;
		_neighborhoodSize = _kernelWidth * _kernelWidth;
		_numFeatureValsPerPoint = NumFeatureValsPerPoint(_featureType, _numDims, _numHistBins, _neighborhoodSize);
	}

    SpidrParameters(size_t numPoints, size_t numDims, ImgSize imgSize, std::string embeddingName, const float* dataVecBegin, size_t numForegroundPoints,
        feature_type featureType, loc_Neigh_Weighting neighWeighting, size_t numLocNeighbors, size_t numHistBins,
        knn_library aknn_algorithm, distance_metric aknn_metric,
        float perplexity, int numIterations, int exaggeration, bool forceCalcBackgroundFeatures = false) :
        _numPoints(numPoints), _numDims(numDims), _imgSize(imgSize), _embeddingName(embeddingName), _numForegroundPoints(numForegroundPoints),
        _featureType(featureType), _neighWeighting(neighWeighting), _numLocNeighbors(numLocNeighbors), _numHistBins(numHistBins),
        _aknn_algorithm(aknn_algorithm), _aknn_metric(aknn_metric), _forceCalcBackgroundFeatures(forceCalcBackgroundFeatures),
        _perplexity(perplexity), _perplexity_multiplier(3), _numIterations(numIterations), _exaggeration(exaggeration)
    {
		update_nn();	// sets nn based on perplexity
		_kernelWidth = (2 * _numLocNeighbors) + 1;
        _neighborhoodSize = _kernelWidth * _kernelWidth;
        _numFeatureValsPerPoint = NumFeatureValsPerPoint(_featureType, _numDims, _numHistBins, _neighborhoodSize);
    }

	// number of nn depends on perplexity, thus the user should not be able to set it
	size_t get_nn() const { return _nn; };
	float get_perplexity() const { return _perplexity; };
	int get_perplexity_multiplier() const { return _perplexity_multiplier; };

	// setting the perplexity also changes the number of knn
	void set_perplexity(float perp) { 
		_perplexity = perp;
		update_nn();	// sets nn based on perplexity
	}

private:
	void update_nn() {
		// see Van Der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms. The Journal of Machine Learning Research, 15(1), 3221-3245.
		_nn = _perplexity * _perplexity_multiplier + 1;

		// For small data sets, use less kNN
		if (_nn > _numPoints)
		{
			spdlog::warn("SpidrParameters: Few data points - reduce number nn to number of points");
			_nn = _numPoints;
		}
	}

public:
	// data
	size_t              _numPoints;             /*!<> */
	size_t              _numDims;               /*!<> */
	ImgSize             _imgSize;               /*!<> */
	std::string         _embeddingName;         /*!< Name of the embedding */
    size_t              _numForegroundPoints;   /*!< _numForegroundPoints - number of background points */
	// features
	feature_type        _featureType;           /*!< */
	size_t              _numFeatureValsPerPoint;/*!< depending on the feature type, the features vector has a different length (scalar features vs vector features per dimension)> */
	loc_Neigh_Weighting _neighWeighting;        /*!<> */
	size_t              _kernelWidth;           /*!< (2 * _numLocNeighbors) + 1;> */
	size_t              _neighborhoodSize;      /*!< _kernelWidth * _kernelWidth> */
	size_t              _numLocNeighbors;       /*!<> */
	size_t              _numHistBins;           /*!<> */
    bool                _forceCalcBackgroundFeatures; /*!<> */
	// distance
	knn_library         _aknn_algorithm;        /*!<> */
	distance_metric     _aknn_metric;           /*!<> */
	// embeddings
	int                 _numIterations;         /*!< Number of gradient descent iterations> */
	int                 _exaggeration;          /*!< Number of iterations for early exageration> */
	int                 _expDecay;              /*!< exponential decay> */

private:
	size_t              _nn;                    // number of nearest neighbors, determined by _perplexity*_perplexity_multiplier + 1
	const int           _perplexity_multiplier; //! Multiplied by the perplexity gives the number of nearest neighbors used
	float               _perplexity;            //! Perplexity value in evert distribution.

};