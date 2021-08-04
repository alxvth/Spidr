#pragma once

#include <cmath>
#include <vector>

#include "SpidrAnalysisParameters.h"
#include "Eigen/Dense"

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
 * \return 
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


/*! Helper struct for constant padding, see padConst
 *  Creates a sequence of indices: padAllDirections{3, 1} -> [0 0 1 2 2]
 *  Thus padding const values like [0 1 2] -> [(0) 0 1 2 (2)]
 *
 * \param in_size 
 * \param pad_size 
 */
struct padAllDirections;


namespace Eigen {
    // add short matrix version for unsigned int, works just as MatrixXi
    typedef Matrix<unsigned int, -1, -1> MatrixXui;
}


/*! Pads a matrix (2d) in all directions with the border values
 *
 * \param mat
 * \param pad_size
 */
Eigen::MatrixXui padConst(Eigen::MatrixXui mat, Eigen::Index pad_size);


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
