#pragma once

#include <cmath>
#include <vector>
#include <utility>  // std::pair
#include <tuple>  // std::tuple
#include <memory>  // std::unique_ptr

#include "SpidrAnalysisParameters.h"
#include <Eigen/Dense>

/*! Normalizes all values in vec wrt to normVal
 * Basically normedVec[i] = vec[i] / normVal
 *
 * \param vec
 * \param normVal
 */
template<typename T>
void NormVector(std::vector<T>& vec, T normVal);

/*!
 *
 * \param n
 * \return
 */
std::vector<unsigned int> PascalsTriangleRow(const unsigned int n);

/*!
 *
 * \param width
 * \param norm
 * \return
 */
std::vector<float> BinomialKernel2D(const unsigned int width, norm_vec norm = norm_vec::NORM_NONE);

/*!
 *
 * \param width
 * \param sd
 * \return
 */
std::vector<float> GaussianKernel1D(const unsigned int width, const float sd = 1);

/*!
 *
 * \param width
 * \param sd
 * \param norm
 * \return
 */
std::vector<float> GaussianKernel2D(const unsigned int width, const float sd = 1, norm_vec norm = norm_vec::NORM_NONE);

/*!
 *
 * \param numItems
 * \return
 */
unsigned int SqrtBinSize(unsigned int numItems);

/*!
 *
 * \param numItems
 * \return
 */
unsigned int SturgesBinSize(unsigned int numItems);

/*!
 *
 * \param numItems
 * \return
 */
unsigned int RiceBinSize(unsigned int numItems);


/*! Get data for all neighborhood point ids
 * Padding: if neighbor is outside selection, assign 0 to all dimension values
 * 
 * \param neighborIDs
 * \param _attribute_data
 * \param _neighborhoodSize
 * \param _numDims
 * \return data layout with dimension d and neighbor n: [n0d0, n0d1, n0d2, ..., n1d0, n1d1, ..., n2d0, n2d1, ...]

 */
std::vector<float> getNeighborhoodValues(const std::vector<int>& neighborIDs, const std::vector<float>& attribute_data, const size_t neighborhoodSize, const size_t numDims);

/*! Calculate the minimum and maximum value for each channel
 *
 * \param numPoints
 * \param numDims
 * \param attribute_data
 * \return vector with [min_Ch0, max_Ch0, min_Ch1, max_Ch1, ...]
 */
template<typename T>
std::vector<float> CalcMinMaxPerChannel(size_t numPoints, size_t numDims, const std::vector<T>& attribute_data) {
    std::vector<float> minMaxVals(2 * numDims, 0);

    // for each dimension iterate over all values
    // remember data stucture (point1 d0, point1 d1,... point1 dn, point2 d0, point2 d1, ...)
    for (unsigned int dimCount = 0; dimCount < numDims; dimCount++) {
        // init min and max
        float currentVal = attribute_data[dimCount];
        minMaxVals[2 * dimCount] = currentVal;
        minMaxVals[2 * dimCount + 1] = currentVal;

        for (unsigned int pointCount = 0; pointCount < numPoints; pointCount++) {
            currentVal = attribute_data[pointCount * numDims + dimCount];
            // min
            if (currentVal < minMaxVals[2 * dimCount])
                minMaxVals[2 * dimCount] = currentVal;
            // max
            else if (currentVal > minMaxVals[2 * dimCount + 1])
                minMaxVals[2 * dimCount + 1] = currentVal;
        }
    }

    return minMaxVals;
}

/*! Calculate the mean value for each channel
 *
 * \param numPoints
 * \param numDims
 * \param attribute_data
 * \return vector with [mean_Ch0, mean_Ch1, ...]
 */
template<typename T>
std::vector<float> CalcMeanPerChannel(size_t numPoints, size_t numDims, const std::vector<T>& attribute_data) {
    std::vector<float> meanVals(numDims, 0);

#pragma omp parallel for 
    for (int dimCount = 0; dimCount < (int)numDims; dimCount++) {
        float sum = 0;
        for (unsigned int pointCount = 0; pointCount < numPoints; pointCount++) {
            sum += attribute_data[pointCount * numDims + dimCount];
        }

        meanVals[dimCount] = sum / numPoints;
    }

    return meanVals;
}

/*! Calculate estimate of the variance
 *  Assuming equally likely values, a (biased) estimated of the variance is computed for each dimension
 *
 * \param numPoints
 * \param numDims
 * \param attribute_data
 * \return vector with [var_Ch0, var_Ch1, ...]
 */
template<typename T>
std::vector<float> CalcVarEstimate(size_t numPoints, size_t numDims, const std::vector<T>& attribute_data, const std::vector<float> &meanVals) {
    std::vector<float> varVals(numDims, 0);

#pragma omp parallel for 
    for (int dimCount = 0; dimCount < (int)numDims; dimCount++) {
        float sum = 0;
        float temp_diff = 0;
        for (unsigned int pointCount = 0; pointCount < numPoints; pointCount++) {
            temp_diff = attribute_data[pointCount * numDims + dimCount] - meanVals[dimCount];
            sum += (temp_diff * temp_diff);
        }

        varVals[dimCount] = (sum > 0) ? sum / numPoints : 0.00000001f;   // make sure that variance is not zero for noise-free data

    }

    return varVals;
}


namespace Eigen {
    // add short matrix version for unsigned int, works just as MatrixXi
	typedef Matrix<unsigned int, -1, -1> MatrixXui;
	//typedef Vector<unsigned int, -1> VectorXui;
}

// What does this do? From https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
// pad{3, 5} creates a sequence of indices [0 0 0 1 2]
// Now a slicing operation A(seqN(i,m), seqN(j,n) selects a block starting at i,j having m rows, and n columns (equivalent to A.block(i,j,m,n)).
// Slicing like A(pad{3,N}, pad{3,N} will thus return a matrix that was padded left and top with 2 rows
struct padUpperLeft {
	Eigen::Index size() const { return out_size; }
	Eigen::Index operator[] (Eigen::Index i) const { return std::max<Eigen::Index>(0, i - (out_size - in_size)); }
	Eigen::Index in_size, out_size;
};

/*! Helper struct for constant padding, see padEdge
 *  Creates a sequence of indices: padAllDirections{3, 1} -> [0 0 1 2 2]
 *  Thus padding const values like [0 1 2] -> [(0) 0 1 2 (2)]
 *
 * \param in_size
 * \param pad_size
 */
struct padAllDirections {
	padAllDirections(Eigen::Index in_size, Eigen::Index pad_size) : in_size(in_size), pad_size(pad_size) {}
	Eigen::Index size() const { return in_size + 2 * pad_size; }
	Eigen::Index operator[] (Eigen::Index i) const { return std::min<Eigen::Index>(std::max<Eigen::Index>(0, i - pad_size), in_size - 1); }
	Eigen::Index in_size, pad_size;
};


/*! Pads a matrix (2d) in all directions with the border values
 * Similar to numpy's np.pad(..., mode='edge')
 *
 * \param mat
 * \param pad_size
 */
Eigen::MatrixXui padEdge(Eigen::MatrixXui mat, Eigen::Index pad_size);

/*! Get rectangle neighborhood point ids for one data item
 *  
 * Padding: constant border value
 * \param coord_row
 * \param coord_col
 * \param kernelWidth
 * \param padded_ids
 * \return
 */
std::vector<int> getNeighborhoodInds(const unsigned int coord_row, const unsigned int coord_col, const size_t kernelWidth, Eigen::MatrixXui* padded_ids);


template <class scalar_type>
class Histogram_Base
{
public:
    Histogram_Base() = delete;
    Histogram_Base(float min, float max, unsigned int numberOfBins);
    // The last bin might be smaller than the rest of (max-min)/binWidth does not yield an integer
    Histogram_Base(float min, float max, float binWidth);

    void fill(const float value);
    void fill(const std::vector<float> values);

    // Getter

    unsigned int getNumBins() const { return _counts.size(); };
    unsigned int getCount(unsigned int bin) const { return _counts[bin]; };

    unsigned int getCount() const { return _countBinValid; };
    unsigned int getCountAll() const { return _countBinTotal; };
    unsigned int getCountUnderflow() const { return _countBinUnderflow; };
    unsigned int getCountOverflow() const { return _countBinOverflow; };

    float getMin() const { return _minVal; };
    float getMax() const { return _maxVal; };
    float getBinLower(unsigned int bin) const { return _minVal + bin * _binWidth; };
    float getBinUpper(unsigned int bin) const { return _minVal + (bin + 1) * _binWidth; };

    auto cbegin() const { return _counts.cbegin(); };
    auto cend() const { return _counts.cend(); };

    scalar_type operator[](int index) const;

    Eigen::Vector<scalar_type, -1> counts() const { return _counts; };
    Eigen::VectorXf normalizedCounts() const { return _counts.cast<float>() / _counts.sum(); };

protected:
    Eigen::Vector<scalar_type, -1> _counts;
    unsigned int _countBinOverflow;
    unsigned int _countBinUnderflow;
    unsigned int _countBinTotal;
    unsigned int _countBinValid;
    float _binWidth;
    float _binNormed;
    float _minVal;
    float _maxVal;
    unsigned int _numBins;

    void commonInit();
};


/*! Histogram class
 *
 * If newVal == binMax then it will not count as overflow but is counted in the largest bin
 */
class Histogram : public Histogram_Base<unsigned int>
{
public:
    Histogram() = delete;
    Histogram(float min, float max, unsigned int numberOfBins) : Histogram_Base(min, max, numberOfBins) { };
    Histogram(float min, float max, float binWidth) : Histogram_Base(min, max, binWidth) { };

};

class Histogram_Weighted : public Histogram_Base<float>
{
public:
    Histogram_Weighted() = delete;
    Histogram_Weighted(float min, float max, unsigned int numberOfBins) : Histogram_Base(min, max, numberOfBins) { };
    Histogram_Weighted(float min, float max, float binWidth) : Histogram_Base(min, max, binWidth) { };

    void fill_weighted(const float value, const float weight);
    void fill_weighted(const std::vector<float> values, const std::vector<float> weights);

};


/*! Base class for channel histograms
 *
 * This histogram class counts active channel values, i.e. one bin is one channel and not a value range
 */
template <class scalar_type>
class Channel_Histogram_Base
{
public:
    Channel_Histogram_Base() = delete;
    Channel_Histogram_Base(size_t numDims, float threshold = 1);
    Channel_Histogram_Base(std::vector<float> tresholds);

    void fill_ch(const size_t ch, const float value);
    void fill_ch(const size_t ch, const std::vector<float> values);

    // Getter

    unsigned int getNumBins() const { return _counts.size(); };
    unsigned int getCount(unsigned int bin) const { return _counts[bin]; };

    unsigned int getCount() const { return _totalBinCounts; };

    auto cbegin() const { return _counts.cbegin(); };
    auto cend() const { return _counts.cend(); };

    scalar_type operator[](size_t index) const;

    Eigen::Vector<scalar_type, -1> counts() const { return _counts; };
    const Eigen::Vector<scalar_type, -1>* countsp() const { return &_counts; };
    Eigen::VectorXf normalizedCounts() const { return _counts.cast<float>() / _counts.sum(); };

    std::vector<scalar_type> counts_std() const { return std::vector<scalar_type>(_counts.data(), _counts.data() + _counts.size()); };
    std::vector<float> normalizedCounts_std() const { auto eigen_counts_norm = normalizedCounts(); return std::vector<scalar_type>(eigen_counts_norm.data(), eigen_counts_norm.data() + eigen_counts_norm.size()); };


protected:
    Eigen::Vector<scalar_type, -1> _counts;

    std::vector<float> _tresholds;

    size_t _totalBinCounts;
    size_t _numBins;

};


class Channel_Histogram : public Channel_Histogram_Base<unsigned int>
{
public:
    Channel_Histogram() = delete;
    Channel_Histogram(unsigned int numDims, float threshold = 1) : Channel_Histogram_Base(numDims, threshold) { };
    Channel_Histogram(std::vector<float> tresholds) : Channel_Histogram_Base(tresholds) { };

};

class Channel_Histogram_Weighted : public Channel_Histogram_Base<float>
{
public:
    Channel_Histogram_Weighted() = delete;
    Channel_Histogram_Weighted(size_t numDims, float threshold = 1) : Channel_Histogram_Base(numDims, threshold) { };
    Channel_Histogram_Weighted(std::vector<float> tresholds) : Channel_Histogram_Base(tresholds) { };

    void fill_ch_weighted(const size_t ch, const float value, const float weight);
    void fill_ch_weighted(const size_t ch, const std::vector<float> values, const std::vector<float> weights);

};

// ####
float variance(Eigen::VectorXf vec);
float covariance(Eigen::VectorXf vec1, Eigen::VectorXf vec2);
Eigen::MatrixXf covmat(Eigen::MatrixXf data);
Eigen::MatrixXf covmat(Eigen::MatrixXf data, Eigen::VectorXf probs);

typedef std::pair<Eigen::VectorXf, Eigen::MatrixXf> multivar_normal;
typedef std::tuple<Eigen::VectorXf, Eigen::MatrixXf, float> multivar_normal_plusDet;

multivar_normal compMultiVarFeatures(Eigen::MatrixXf data);
multivar_normal compMultiVarFeatures(Eigen::MatrixXf data, Eigen::VectorXf probs);


class IFeatureData
{
};

template<class T>
class FeatureData : public IFeatureData
{
public:
    FeatureData(T d) : data(d) {};
    T data;
};


class Feature
{
public:
    Feature() { };
    ~Feature() { }

    //std::vector<std::unique_ptr<IFeatureData>>* get_data_ptru() { return &featdata_u; };
    std::vector<IFeatureData*>* get_data_ptr() { return &featdata; };
    void resize(size_t newSize) { featdata.resize(newSize); };

    IFeatureData* at(size_t ID) const { return featdata.at(ID); };

    template<class T>
    T cast_at(size_t ID) { return static_cast<T>(featdata.at(ID)); };

protected:
    //std::vector<std::unique_ptr<IFeatureData>> featdata_u;
    std::vector<IFeatureData*> featdata;
};
