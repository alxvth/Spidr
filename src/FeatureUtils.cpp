#include "FeatureUtils.h"

#include <execution>    // par_unseq
#include <algorithm>    // for_each_n
#include <numeric>      // iota
#include <cmath> 
#include <stdexcept>


template<typename T>
void NormVector(std::vector<T>& vec, T normVal) {

    std::for_each(std::execution::par_unseq, std::begin(vec), std::end(vec), [normVal](auto& val) {
        val /= normVal;
    });

}

std::vector<unsigned int> PascalsTriangleRow(const unsigned int n) {
    std::vector<unsigned int> row(n + 1, 1);
    unsigned int entry = 1;
    for (unsigned int i = 1; i < n + 1; i++) {
        entry = (unsigned int)(entry * (n + 1 - i) / i);
        row[i] = entry;
    }
    return row;
}

// @param norm: 1 indicates max, 2 indicates sum, 0 indicates no normalization
std::vector<float> BinomialKernel2D(const unsigned int width, norm_vec norm) {
    if (width % 2 == 0)
        throw std::invalid_argument("n must be odd");

    std::vector<unsigned int> bino1D = PascalsTriangleRow(width - 1);
    std::vector<float> bino2D(width * width, -1);

    // helper for normalization
    int sum = 0;
    int max = 0;

    // outter product
    for (unsigned int row = 0; row < width; row++) {
        for (unsigned int col = 0; col < width; col++) {
            bino2D[row*width + col] = bino1D[row] * bino1D[col];

            // helper for normalization
            sum += +bino2D[row*width + col];
            if (bino2D[row*width + col] > (float)max)
                max = bino2D[row*width + col];
        }
    }

    // normalization
    if (norm == norm_vec::NORM_MAX)
        NormVector(bino2D, (float)max);
    else if (norm == norm_vec::NORM_SUM)
        NormVector(bino2D, (float)sum);

    return bino2D;
}

std::vector<float> GaussianKernel1D(const unsigned int width, const float sd) {
    if (width % 2 == 0)
        throw std::invalid_argument("n must be odd");
    if (sd < 0)
        throw std::invalid_argument("sd must be positive");

    std::vector<float> kernel(width, 0);
    int coutner = 0;
    for (int i = (-1 * ((int)width - 1) / 2); i <= ((int)width - 1) / 2; i++) {
        kernel[coutner] = std::exp(-1 * (i*i) / (2 * sd * sd));
        coutner++;
    }
    return kernel;

}

// @param norm: 1 indicates max, 2 indicates sum, 0 indicates no normalization
std::vector<float> GaussianKernel2D(const unsigned int width, const float sd, norm_vec norm) {
    if (width % 2 == 0)
        throw std::invalid_argument("n must be odd");
    if (sd < 0)
        throw std::invalid_argument("sd must be positive");

    std::vector<float> gauss1D = GaussianKernel1D(width);
    std::vector<float> gauss2D(width * width, -1);

    // helper for normalization
    float sum = 0;
    float max = 0;

    // outter product
    for (unsigned int row = 0; row < width; row++) {
        for (unsigned int col = 0; col < width; col++) {
            gauss2D[row*width + col] = gauss1D[row] *  gauss1D[col];

            // helper for normalization
            sum += +gauss2D[row*width + col];
            if (gauss2D[row*width + col] > (float)max)
                max = gauss2D[row*width + col];
        }
    }

    // normalization
    if (norm == norm_vec::NORM_MAX)
        NormVector(gauss2D, max);
    else if (norm == norm_vec::NORM_SUM)
        NormVector(gauss2D, sum);

    return gauss2D;
}

unsigned int SqrtBinSize(unsigned int numItems) {
    return int(std::ceil(std::sqrt(numItems)));
}

unsigned int SturgesBinSize(unsigned int numItems) {
    return int(std::ceil(std::log2(numItems) + 1));
}

unsigned int RiceBinSize(unsigned int numItems) {
    return int(std::ceil((2 * std::pow(numItems, 1.0/3))));
}

std::vector<float> getNeighborhoodValues(const std::vector<int>& neighborIDs, const std::vector<float>& attribute_data, const size_t neighborhoodSize, const size_t numDims) {
    std::vector<float> neighborValues(neighborhoodSize * numDims);
#ifdef NDEBUG
    // later an assert can check whether all values are different from FLT_MAX
    std::fill(neighborValues.begin(), neighborValues.end(), FLT_MAX);
#endif

    // data layout with dimension d and neighbor n: [n0d0, p0d1, n0d2, ..., n1d0, n1d1, ..., n2d0, n2d1, ...]
    for (unsigned int neighbor = 0; neighbor < neighborhoodSize; neighbor++) {
        for (unsigned int dim = 0; dim < numDims; dim++) {
            neighborValues[neighbor * numDims + dim] = attribute_data[neighborIDs[neighbor] * numDims + dim];
        }
    }
    return neighborValues;
}


std::vector<int> getNeighborhoodInds(const unsigned int coord_row, const unsigned int coord_col, const size_t kernelWidth, Eigen::MatrixXui* padded_ids) {
    Eigen::MatrixXui neighborhoodInds_mat = padded_ids->block(coord_row, coord_col, kernelWidth, kernelWidth);
    std::vector<int> neighborhoodInds_vec(neighborhoodInds_mat.data(), neighborhoodInds_mat.data() + neighborhoodInds_mat.size());
    return neighborhoodInds_vec;
}


Eigen::MatrixXui padConst(Eigen::MatrixXui mat, Eigen::Index pad_size)
{
	//auto slice_sequence_rows = padAllDirections{ mat.rows(), pad_size };
	//auto slice_sequence_cols = padAllDirections{ mat.cols(), pad_size };
	//auto padded_mat = mat(slice_sequence_rows, slice_sequence_cols)
	return mat(padAllDirections{ mat.rows(), pad_size }, padAllDirections{ mat.cols(), pad_size });
}


template< class scalar_type> Histogram_Base< scalar_type>::Histogram_Base(float min, float max, unsigned int numberOfBins) :
    _minVal(min), _maxVal(max), _numBins(numberOfBins), _countBinTotal(0), _countBinValid(0), _countBinUnderflow(0), _countBinOverflow(0)
{
    _binWidth = (_maxVal - _minVal) / (float)_numBins;
    commonInit();
}
// Resolve linker errors with explicit instantiation, https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
template Histogram_Base<unsigned int>::Histogram_Base(float min, float max, unsigned int numberOfBins);
template Histogram_Base<float>::Histogram_Base(float min, float max, unsigned int numberOfBins);

template< class scalar_type> Histogram_Base< scalar_type>::Histogram_Base(float min, float max, float binWidth) :
    _minVal(min), _maxVal(max), _binWidth(binWidth), _countBinTotal(0), _countBinValid(0), _countBinUnderflow(0), _countBinOverflow(0)
{
    _numBins = std::ceil((_maxVal - _minVal) / (float)_binWidth);
    commonInit();
}
template Histogram_Base<unsigned int>::Histogram_Base(float min, float max, float binWidth);
template Histogram_Base<float>::Histogram_Base(float min, float max, float binWidth);


template< class scalar_type> void Histogram_Base< scalar_type>::commonInit()
{
    if (_minVal >= _maxVal)
        throw std::runtime_error("Histogram_Base: Bin max must be larger than bin min.");

    _counts = Eigen::Vector<scalar_type, -1>::Zero(_numBins);
    _binNormed = (float)_numBins / (_maxVal - _minVal);
}
template void Histogram_Base<unsigned int>::commonInit();
template void Histogram_Base<float>::commonInit();


template< class scalar_type> void Histogram_Base< scalar_type>::fill(const float value) {
    unsigned int binID;
    if (value >= _minVal && value < _maxVal) {
        binID = std::floor((value - _minVal) * _binNormed);
        _counts[binID] += 1;
        _countBinValid += 1;
    }
    else if (value == _maxVal)
    {
        _counts[_numBins - 1] += 1;
        _countBinValid += 1;
    }
    else if (value > _maxVal) {
        _countBinOverflow += 1;
    }
    else {
        _countBinUnderflow += 1;
    }

    _countBinTotal += 1;
}
template void Histogram_Base<unsigned int>::fill(const float value);
template void Histogram_Base<float>::fill(const float value);

template< class scalar_type> void Histogram_Base< scalar_type>::fill(const std::vector<float> values) {
    for (const float &value : values)
        fill(value);
}
template void Histogram_Base<unsigned int>::fill(const std::vector<float> values);
template void Histogram_Base<float>::fill(const std::vector<float> values);


template< class scalar_type> scalar_type Histogram_Base< scalar_type>::operator[](int index) const {
    assert(index >= 0 && index < _numBins);
    return _counts[index];
}
template unsigned int Histogram_Base<unsigned int>::operator[](int index) const;
template float Histogram_Base<float>::operator[](int index) const;


void Histogram_Weighted::fill_weighted(const float value, const float weight) {
    unsigned int binID;
    if (value >= _minVal && value < _maxVal) {
        binID = std::floor((value - _minVal) * _binNormed);
        _counts[binID] += weight;
        _countBinValid += 1;
    }
    else if (value == _maxVal)
    {
        _counts[_numBins - 1] += weight;
        _countBinValid += 1;
    }
    else if (value > _maxVal) {
        _countBinOverflow += 1;
    }
    else {
        _countBinUnderflow += 1;
    }

    _countBinTotal += 1;
}

void Histogram_Weighted::fill_weighted(const std::vector<float> values, const std::vector<float> weights) {
    assert(values.size() == weights.size());

    for (unsigned int i = 0; i < values.size(); i++)
        fill_weighted(values[i], weights[i]);
}



float variance(Eigen::VectorXf vec)
{
    Eigen::VectorXf centered = vec.array() - vec.mean();
    return (1.0f / vec.size()) * centered.dot(centered);
}

float covariance(Eigen::VectorXf vec1, Eigen::VectorXf vec2)
{
    assert(vec1.size() == vec2.size());
    Eigen::VectorXf centered1 = vec1.array() - vec1.mean();
    Eigen::VectorXf centered2 = vec2.array() - vec2.mean();
    return (1.0f / vec1.size()) * centered1.dot(centered2);
}

Eigen::MatrixXf covmat(Eigen::MatrixXf data)
{
    // see https://stackoverflow.com/a/15142446 and https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Definition_of_sample_covariance
    // also https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Unbiasedness for discussion over (1 / (neighborhood.cols() - 1)) and (1 / neighborhood.cols())
    // Since we know all the neighrborhood, we go for (1 / neighborhood.cols())
    Eigen::MatrixXf centered = data.colwise() - data.rowwise().mean();
    return (1.0f / data.cols()) * (centered * centered.transpose());
}

Eigen::MatrixXf covmat(Eigen::MatrixXf data, Eigen::VectorXf probs)
{
    assert(probs.size() == data.cols());			// one prob value for each observation
    assert(std::abs(probs.sum() - 1.0f) < 0.001);  // cumulative prob must be 1

    // see https://stackoverflow.com/a/15142446 and https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Definition_of_sample_covariance
    // also https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Unbiasedness for why it's (1 / (neighborhood.cols() - 1)) and not (1 / neighborhood.cols())
    // see https://stackoverflow.com/a/42945996 for centered * probs.asDiagonal() 
    Eigen::MatrixXf centered = data.colwise() - data.rowwise().mean();
    return ((centered * probs.asDiagonal()) * centered.transpose());

    // just for me to check that they are actually identical
    // std::cout << "test 1: \n" << neighborhood.array().rowwise() * probs.transpose().array() << "\n";
    // std::cout << "test 2: \n" << neighborhood * probs.asDiagonal() << "\n";

}

multivar_normal compMultiVarFeatures(Eigen::MatrixXf data) {
    Eigen::VectorXf mean = data.rowwise().mean();
    Eigen::MatrixXf centered = data.colwise() - mean;
    Eigen::MatrixXf cov_mat = (centered * centered.transpose()) / data.cols();
    return std::make_pair(mean, cov_mat);
}

multivar_normal compMultiVarFeatures(Eigen::MatrixXf data, Eigen::VectorXf probs) {
    Eigen::VectorXf mean = data.rowwise().mean();
    Eigen::MatrixXf centered = data.colwise() - mean;
    Eigen::MatrixXf cov_mat = ((centered * probs.asDiagonal()) * centered.transpose());
    return std::make_pair(mean, cov_mat);
}
