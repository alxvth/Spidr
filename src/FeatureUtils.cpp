#include "FeatureUtils.h"

#include <execution>    // par_unseq
#include <algorithm>    // for_each_n
#include <numeric>      // iota
#include <iostream>      // iota


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

    for (unsigned int neighbor = 0; neighbor < neighborhoodSize; neighbor++) {
        for (unsigned int dim = 0; dim < numDims; dim++) {
            neighborValues[neighbor * numDims + dim] = attribute_data[neighborIDs[neighbor] * numDims + dim];
        }
    }
    return neighborValues;
}

struct padAllDirections {
    padAllDirections(Eigen::Index in_size, Eigen::Index pad_size) : in_size(in_size), pad_size(pad_size) {}
    Eigen::Index size() const { return in_size + 2 * pad_size; }
    Eigen::Index operator[] (Eigen::Index i) const { return std::min<Eigen::Index>(std::max<Eigen::Index>(0, i - pad_size), in_size - 1); }
    Eigen::Index in_size, pad_size;
};


Eigen::MatrixXui padConst(Eigen::MatrixXui mat, Eigen::Index pad_size)
{
    // auto slice_sequence_rows = padAllDirections{ mat.rows(), pad_size };
    // auto slice_sequence_cols = padAllDirections{ mat.cols(), pad_size };
    // auto padded_mat = mat(slice_sequence_rows, slice_sequence_cols)

    return mat(padAllDirections{ mat.rows(), pad_size }, padAllDirections{ mat.cols(), pad_size });
}

std::vector<int> getNeighborhoodInds(const unsigned int coord_row, const unsigned int coord_col, const size_t kernelWidth, Eigen::MatrixXui* padded_ids) {
    Eigen::MatrixXui neighborhoodInds_mat = padded_ids->block(coord_row, coord_col, kernelWidth, kernelWidth);
    //std::cout << neighborhoodInds_mat << "\n" << std::endl;
    std::vector<int> neighborhoodInds_vec(neighborhoodInds_mat.data(), neighborhoodInds_mat.data() + neighborhoodInds_mat.size());
    return neighborhoodInds_vec;
}
