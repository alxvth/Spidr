#pragma once
#include "hnswlib/hnswlib.h"    // defines USE_SSE and USE_AVX and includes intrinsics

#if defined(__GNUC__)
#define PORTABLE_ALIGN32hnsw __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32hnsw __declspec(align(32))
#endif

#include <omp.h>

#include <cmath>     // std::sqrt, exp, floor
#include <numeric>   // std::inner_product, std:accumulate 
#include <algorithm> // std::find, fill, sort
#include <vector>
#include <thread>
#include <atomic>

#include "hdi/data/map_mem_eff.h" // hdi::data::MapMemEff

#include <chrono>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "SpidrAnalysisParameters.h"
#include "FeatureUtils.h"

namespace hnswlib {


    /* !
     * The method is borrowed from nmslib, https://github.com/nmslib/nmslib/blob/master/similarity_search/include/thread_pool.h
     */
    template<class Function>
    inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
        if (numThreads <= 0) {
            numThreads = std::thread::hardware_concurrency();
        }

        if (numThreads == 1) {
            for (size_t id = start; id < end; id++) {
                fn(id, 0);
            }
        }
        else {
            std::vector<std::thread> threads;
            std::atomic<size_t> current(start);

            // keep track of exceptions in threads
            // https://stackoverflow.com/a/32428427/1713196
            std::exception_ptr lastException = nullptr;
            std::mutex lastExceptMutex;

            for (size_t threadId = 0; threadId < numThreads; ++threadId) {
                threads.push_back(std::thread([&, threadId] {
                    while (true) {
                        size_t id = current.fetch_add(1);

                        if ((id >= end)) {
                            break;
                        }

                        try {
                            fn(id, threadId);
                        }
                        catch (...) {
                            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                            lastException = std::current_exception();
                            /*
                             * This will work even when current is the largest value that
                             * size_t can fit, because fetch_add returns the previous value
                             * before the increment (what will result in overflow
                             * and produce 0 instead of current + 1).
                             */
                            current = end;
                            break;
                        }
                    }
                }));
            }
            for (auto &thread : threads) {
                thread.join();
            }
            if (lastException) {
                std::rethrow_exception(lastException);
            }
        }

    }


    // ---------------
    // Quadratic form for 1D Histograms
    // ---------------

    // data struct for distance calculation in QFSpace
    struct space_params_QF {
        size_t dim;
        size_t bin;
        ::std::vector<float> A;     // bin similarity matrix for 1D histograms: entry A_ij refers to the sim between bin i and bin j 
        Eigen::MatrixXf weights;    // same as A
    };

    static float
        QFSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        FeatureData<std::vector<Eigen::VectorXf>>* histos1 = static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect1v);
        FeatureData<std::vector<Eigen::VectorXf>>* histos2 = static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect2v);

        //float* pVect1 = (float*)pVect1v;
        //float* pVect2 = (float*)pVect2v;

        const space_params_QF* sparam = (space_params_QF*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t nbin = sparam->bin;
        const float* pWeight = sparam->A.data();

        float res = 0;
        float t1 = 0;
        float t2 = 0;

        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            // QF distance = sum_ij ( a_ij * (x_i-y_i) * (x_j-y_j) )
            for (size_t i = 0; i < nbin; i++) {
                t1 = histos1->data[d][i] - histos2->data[d][i];
                for (size_t j = 0; j < nbin; j++) {
                    t2 = histos1->data[d][j] - histos2->data[d][j];
                    res += *(pWeight + i * nbin + j) * t1 * t2;
                }
            }
            // point to next dimension
            //pVect1 += nbin;
            //pVect2 += nbin;
        }

        return res;
    }

    // This one is much slower
    //static float
    //    QFEigenSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    //    Eigen::VectorXf* histos1 = (static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect1v)->data).data();
    //    Eigen::VectorXf* histos2 = (static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect2v)->data).data();

    //    const space_params_QF* sparam = (space_params_QF*)qty_ptr;
    //    const size_t ndim = sparam->dim;
    //    const size_t nbin = sparam->bin;
    //    const Eigen::MatrixXf weights = sparam->weights;

    //    float res = 0;
    //    float t1 = 0;
    //    float t2 = 0;

    //    Eigen::VectorXf diff;

    //    // add the histogram distance for each dimension
    //    for (size_t d = 0; d < ndim; d++) {
    //        // QF distance = sum_ij ( a_ij * (x_i-y_i) * (x_j-y_j) )

    //        diff = *histos1 - *histos2;
    //        res += diff.transpose() * weights * diff;
    //        
    //        // point to histograms of next dimension
    //        histos1++;
    //        histos2++;
    //    }

    //    return res;
    //}

    static float
        QFSqrSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        std::vector<Eigen::VectorXf>* histos1 = &(static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect1v)->data);
        std::vector<Eigen::VectorXf>* histos2 = &(static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect2v)->data);

        float* pVect1 = nullptr;
        float* pVect2 = nullptr;

        space_params_QF* sparam = (space_params_QF*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t nbin = sparam->bin;

        size_t nbin4 = nbin >> 2 << 2;		// right shift by 2, left-shift by 2: create a multiple of 4

        float res = 0;
        float PORTABLE_ALIGN32hnsw TmpRes[8];			// memory aligned float array
        __m128 v1, v2, TmpSum, wRow, diff;			// write in registers of 128 bit size
        float *pA, *pEnd1, *pW, *pWend, *pwR;
        unsigned int wloc;

        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            pA = sparam->A.data();					// reset to first weight for every dimension

           // calculate the QF distance for each dimension
            pVect1 = (histos1->at(d)).data();
            pVect2 = (histos2->at(d)).data();

           // 1. calculate w = (pVect1-pVect2)
            std::vector<float> w(nbin);
            wloc = 0;
            pEnd1 = pVect1 + nbin4;			// point to the first dimension not to be vectorized
            while (pVect1 < pEnd1) {
                v1 = _mm_loadu_ps(pVect1);					// Load the next four float values
                v2 = _mm_loadu_ps(pVect2);
                diff = _mm_sub_ps(v1, v2);					// substract all float values
                _mm_store_ps(&w[wloc], diff);				// store diff values in memory
                pVect1 += 4;								// advance pointer to position after loaded values
                pVect2 += 4;
                wloc += 4;
            }

            // manually calc the rest dims
            for (wloc; wloc < nbin; wloc++) {
                w[wloc] = *pVect1 - *pVect2;
                pVect1++;
                pVect2++;
            }

            // 2. calculate d = w'Aw
            for (unsigned int row = 0; row < nbin; row++) {
                TmpSum = _mm_set1_ps(0);
                pW = w.data();					// pointer to first float in w
                pWend = pW + nbin4;			// point to the first dimension not to be vectorized
                pwR = pW + row;
                wRow = _mm_load1_ps(pwR);					// load one float into all elements fo wRow

                while (pW < pWend) {
                    v1 = _mm_loadu_ps(pW);
                    v2 = _mm_loadu_ps(pA);
                    TmpSum = _mm_add_ps(TmpSum, _mm_mul_ps(wRow, _mm_mul_ps(v1, v2)));	// multiply all values and add them to temp sum values
                    pW += 4;
                    pA += 4;
                }
                _mm_store_ps(TmpRes, TmpSum);
                res += TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

                // manually calc the rest dims
                for (unsigned int uloc = nbin4; uloc < nbin; uloc++) {
                    res += *pwR * *pW * *pA;
                    pW++;
                    pA++;
                }
            }

            // point to next dimension is done in the last iteration
            // of the for loop in the rest calc under point 1. (no pVect1++ necessary here)
        }

        return res;
    }


    class QFSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        space_params_QF params_;

    public:
        QFSpace(size_t dim, size_t bin, size_t featureValsPerPoint, bin_sim ground_type = bin_sim::SIM_EUC) {

            fstdistfunc_ = QFSqr;
            // Not entirely sure why this only shows positive effects for high bin counts...
            if (bin >= 12)
            {
                fstdistfunc_ = QFSqrSSE;
            }

            //data_size_ = featureValsPerPoint * sizeof(float);
            data_size_ = sizeof(std::vector<Eigen::VectorXf>);

            ::std::vector<float> A = BinSimilarities(bin, ground_type);
            
            Eigen::MatrixXf weights = Eigen::Map<Eigen::MatrixXf>(&A[0], bin, bin);;

            std::cout << weights << "\n";

            params_ = { dim, bin, A, weights };
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return (void *) &params_;
        }

        ~QFSpace() {}
    };
       
    // ---------------
    //    Hellinger
    // ---------------

    // data struct for distance calculation in HellingerSpace
    struct space_params_Hel {
        size_t dim;
        size_t bin;
    };

    static float
        HelSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        FeatureData<std::vector<Eigen::VectorXf>>* histos1 = static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect1v);
        FeatureData<std::vector<Eigen::VectorXf>>* histos2 = static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect2v);

        float *pVect1 = (float *)pVect1v;
        float *pVect2 = (float *)pVect2v;
       
        const space_params_Hel* sparam = (space_params_Hel*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t nbin = sparam->bin;

        float res = 0;

        // Calculate Hellinger distance based on Bhattacharyya coefficient 
        float binSim = 0;
        float histDiff = 1;
        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            histDiff = 1;
            for (size_t b = 0; b < nbin; b++) {
                binSim = histos1->data[d][b] * histos2->data[d][b];
                histDiff -= ::std::sqrt(binSim);
            }
            res += (histDiff>=0) ? ::std::sqrt(histDiff) : 0; // sometimes histDiff is slightly below 0 due to rounding errors
        }

        return (res);
    }


    class HellingerSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Hel params_;

    public:
        HellingerSpace(size_t dim, size_t bin, size_t featureValsPerPoint) {

            fstdistfunc_ = HelSqr;
            params_ = { dim, bin };
            //data_size_ = featureValsPerPoint * sizeof(float);
            data_size_ = sizeof(std::vector<Eigen::VectorXf>);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return (void *) &params_;
        }

        ~HellingerSpace() {}
    };


    // ---------------
    //    Adapt L2 space
    // ---------------

    struct space_params_L2Feat {
        size_t dim;
        DISTFUNC<float> L2distfunc_;
    };


    static float
        L2FeatSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        //FeatureData<std::vector<float>>* histos1 = static_cast<FeatureData<std::vector<float>>*>((IFeatureData*)pVect1v);
        //FeatureData<std::vector<float>>* histos2 = static_cast<FeatureData<std::vector<float>>*>((IFeatureData*)pVect2v);
        float *pVect1 = (static_cast<FeatureData<std::vector<float>>*>((IFeatureData*)pVect1v)->data).data();
        float *pVect2 = (static_cast<FeatureData<std::vector<float>>*>((IFeatureData*)pVect2v)->data).data();

        const space_params_L2Feat* sparam = (space_params_L2Feat*)qty_ptr;
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        return L2distfunc_(pVect1, pVect2, &(sparam->dim));
    }


    class L2FeatSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

        space_params_L2Feat params_;

    public:
        L2FeatSpace(size_t dim) {
            fstdistfunc_ = L2FeatSqr;

            dim_ = dim;
            //data_size_ = dim * sizeof(float);
            data_size_ = sizeof(std::vector<float>);

            params_ = { dim, L2Sqr };

#if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                params_.L2distfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                params_.L2distfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif
       
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &params_;
        }

        ~L2FeatSpace() {}
    };


    // ---------------
    //    Point cloud distance (Chamfer)
    // ---------------

    // data struct for distance calculation in ChamferSpace
    struct space_params_Chamf {
        size_t dim;
        Eigen::VectorXf weights;         // neighborhood similarity matrix
        size_t neighborhoodSize;
        DISTFUNC<float> L2distfunc_;
    };

    static float
        ChamferDist(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const float* valsN1Begin = static_cast<FeatureData<Eigen::MatrixXf>*>((IFeatureData*)pVect1v)->data.data(); // pointer to vector with values in neighborhood 1
        const float* valsN2Begin = static_cast<FeatureData<Eigen::MatrixXf>*>((IFeatureData*)pVect2v)->data.data();

        // parameters
        const space_params_Chamf* sparam = (space_params_Chamf*)qty_ptr;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        const size_t ndim = sparam->dim;
        const Eigen::VectorXf weights = sparam->weights;
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        //Eigen::VectorXf colDistMins(valsN1.cols()); // (2 * (params._numLocNeighbors) + 1) * (2 * (params._numLocNeighbors) + 1)
        //Eigen::VectorXf rowDistMins(valsN1.cols());

        Eigen::MatrixXf distMat(neighborhoodSize, neighborhoodSize);

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2 (colDist) and vice versa (rowDist)
        // Weight the colDist and rowDist with the inverse of the number of items in the neighborhood
        for (int n1 = 0; n1 < neighborhoodSize; n1++) {
            for (int n2 = 0; n2 < neighborhoodSize; n2++) {
                distMat(n1, n2) = L2distfunc_(valsN1Begin + (n1*ndim), valsN2Begin + (n2*ndim), &ndim);
            }
        }
        // Using the SSE function from HSNW is faster than then the matrix version from Eigen
        // There is probably a smart formulation with the Eigen matrices that would be better though...
        // not faster:
        //for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {
        //    distMat.col(n1) = (valsN1.colwise() - valsN2.col(n1)).colwise().squaredNorm();
        //}

        // weight min of each col and row, and sum over them
        //colDistMins = distMat.rowwise().minCoeff();
        //rowDistMins = distMat.colwise().minCoeff();

        //return colSum / numNeighbors1 + rowSum / numNeighbors2;
        return (distMat.rowwise().minCoeff().dot(weights) + distMat.colwise().minCoeff().dot(weights)) / neighborhoodSize;
    }


    class ChamferSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Chamf params_;

    public:
        ChamferSpace(size_t dim, size_t neighborhoodSize, loc_Neigh_Weighting weighting, const float* dataVectorBegin, size_t featureValsPerPoint) {
            fstdistfunc_ = ChamferDist;
            //data_size_ = featureValsPerPoint * sizeof(float);
            data_size_ = sizeof(FeatureData<Eigen::MatrixXf>);

            assert((::std::sqrt(neighborhoodSize) - std::floor(::std::sqrt(neighborhoodSize))) == 0);  // neighborhoodSize must be perfect square
            unsigned int _kernelWidth = (int)::std::sqrt(neighborhoodSize);

            ::std::vector<float> A(neighborhoodSize);
            switch (weighting)
            {
            case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(A.begin(), A.end(), 1); break;
            case loc_Neigh_Weighting::WEIGHT_BINO: A = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;        // weight the center with 1
            case loc_Neigh_Weighting::WEIGHT_GAUS: A = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NONE); break;
            default:  std::fill(A.begin(), A.end(), -1);  break;  // no implemented weighting type given. 
            }

            Eigen::VectorXf weights = Eigen::Map<Eigen::VectorXf>(&A[0], neighborhoodSize);;

            params_ = { dim, weights, neighborhoodSize, L2Sqr };

#if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                params_.L2distfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                params_.L2distfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &params_;
        }

        ~ChamferSpace() {}
    };

//    // ---------------
//    //    Point cloud distance (Chamfer)
//    // ---------------
//
//    // data struct for distance calculation in ChamferSpace
//    struct space_params_Chamf {
//        const float* dataVectorBegin;
//        size_t dim;
//        ::std::vector<float> A;         // neighborhood similarity matrix
//        size_t neighborhoodSize;        //  (2 * (params._numLocNeighbors) + 1) * (2 * (params._numLocNeighbors) + 1)
//        DISTFUNC<float> L2distfunc_;
//    };
//
//    static float
//        ChamferDist(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//        // TODO: switch from IDs to actual values, maybe?
//        std::vector<int> idsN1 = static_cast<FeatureData<std::vector<int>>*>((IFeatureData*)pVect1v)->data; // vector with IDs in neighborhood 1
//        std::vector<int> idsN2 = static_cast<FeatureData<std::vector<int>>*>((IFeatureData*)pVect2v)->data;
//
//        // parameters
//        const space_params_Chamf* sparam = (space_params_Chamf*)qty_ptr;
//        const size_t ndim = sparam->dim;
//        const size_t neighborhoodSize = sparam->neighborhoodSize;
//        const float* dataVectorBegin = sparam->dataVectorBegin;
//        const std::vector<float> weights = sparam->A;
//        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;
//
//        float colSum = 0;
//        float rowSum = 0;
//        std::vector<float> colDistMins(neighborhoodSize, FLT_MAX);
//        std::vector<float> rowDistMins(neighborhoodSize, FLT_MAX);
//        float distN1N2 = 0;
//
//        // Euclidean dist between all neighbor pairs
//        // Take the min of all dists from a item in neigh1 to all items in Neigh2 (colDist) and vice versa (rowDist)
//        // Weight the colDist and rowDist with the inverse of the number of items in the neighborhood
//        for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {
//
//            for (size_t n2 = 0; n2 < neighborhoodSize; n2++) {
//
//                distN1N2 = L2distfunc_(dataVectorBegin + (idsN1[n1] * ndim), dataVectorBegin + (idsN2[n2] * ndim), &ndim);
//
//                if (distN1N2 < colDistMins[n1])
//                    colDistMins[n1] = distN1N2;
//
//                if (distN1N2 < rowDistMins[n2])
//                    rowDistMins[n2] = distN1N2;
//
//            }
//        }
//
//        // weight min of each col and row, and sum over them
//        for (size_t n = 0; n < neighborhoodSize; n++) {
//
//            colSum += colDistMins[n] * weights[n];
//            rowSum += rowDistMins[n] * weights[n];
//
//        }
//
//        assert(colSum < FLT_MAX);
//        assert(rowSum < FLT_MAX);
//
//        //return colSum / numNeighbors1 + rowSum / numNeighbors2;
//        return (colSum + rowSum) / neighborhoodSize;
//    }
//
//
//    class ChamferSpace : public SpaceInterface<float> {
//
//        DISTFUNC<float> fstdistfunc_;
//        size_t data_size_;
//
//        space_params_Chamf params_;
//
//    public:
//        ChamferSpace(size_t dim, size_t neighborhoodSize, loc_Neigh_Weighting weighting, const float* dataVectorBegin, size_t featureValsPerPoint) {
//            fstdistfunc_ = ChamferDist;
//            //data_size_ = featureValsPerPoint * sizeof(float);
//            data_size_ = sizeof(FeatureData<std::vector<int>>);
//
//            assert((::std::sqrt(neighborhoodSize) - std::floor(::std::sqrt(neighborhoodSize))) == 0);  // neighborhoodSize must be perfect square
//            unsigned int _kernelWidth = (int)::std::sqrt(neighborhoodSize);
//
//            ::std::vector<float> A(neighborhoodSize);
//            switch (weighting)
//            {
//            case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(A.begin(), A.end(), 1); break;
//            case loc_Neigh_Weighting::WEIGHT_BINO: A = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;        // weight the center with 1
//            case loc_Neigh_Weighting::WEIGHT_GAUS: A = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NONE); break;
//            default:  std::fill(A.begin(), A.end(), -1);  break;  // no implemented weighting type given. 
//            }
//
//            params_ = { dataVectorBegin, dim, A, neighborhoodSize, L2Sqr };
//
//#if defined(USE_SSE) || defined(USE_AVX)
//            if (dim % 16 == 0)
//                params_.L2distfunc_ = L2SqrSIMD16Ext;
//            else if (dim % 4 == 0)
//                params_.L2distfunc_ = L2SqrSIMD4Ext;
//            else if (dim > 16)
//                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
//            else if (dim > 4)
//                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
//#endif
//        }
//
//        size_t get_data_size() {
//            return data_size_;
//        }
//
//        DISTFUNC<float> get_dist_func() {
//            return fstdistfunc_;
//        }
//
//        void *get_dist_func_param() {
//            return &params_;
//        }
//
//        ~ChamferSpace() {}
//    };

    // ---------------
    //    Point cloud distance (Sum of squared distances)
    // ---------------

// data struct for distance calculation in SSDSpace
    struct space_params_SSD {
        const float* dataVectorBegin;
        size_t dim;
        ::std::vector<float> A;         // neighborhood similarity matrix
        size_t neighborhoodSize;        //  (2 * (params._numLocNeighbors) + 1) * (2 * (params._numLocNeighbors) + 1)
        DISTFUNC<float> L2distfunc_;
    };


    static float
        SumSquaredDist(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        // TODO: switch from IDs to actual values, maybe?
        std::vector<int> idsN1 = static_cast<FeatureData<std::vector<int>>*>((IFeatureData*)pVect1v)->data; // vector with IDs in neighborhood 1
        std::vector<int> idsN2 = static_cast<FeatureData<std::vector<int>>*>((IFeatureData*)pVect2v)->data;

        // parameters
        space_params_SSD* sparam = (space_params_SSD*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        const float* dataVectorBegin = sparam->dataVectorBegin;
        const std::vector<float> weights = sparam->A;
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        float tmpRes = 0;

        int numNeighbors1 = neighborhoodSize - std::count(idsN1.begin(), idsN1.end(), -2.0f);
        int numNeighbors2 = neighborhoodSize - std::count(idsN2.begin(), idsN2.end(), -2.0f);

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2 (colDist) and vice versa (rowDist)
        // Weight the colDist and rowDist with the inverse of the number of items in the neighborhood
        for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {

            for (size_t n2 = 0; n2 < neighborhoodSize; n2++) {

				tmpRes += (weights[n1] + weights[n2]) * L2distfunc_(dataVectorBegin + (idsN1[n1] * ndim), dataVectorBegin + (idsN2[n2] * ndim), &ndim);

            }
        }

        return tmpRes / (numNeighbors1 * numNeighbors2); 
    }


    class SSDSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_SSD params_;

    public:
        SSDSpace(size_t dim, size_t neighborhoodSize, loc_Neigh_Weighting weighting, const float* dataVectorBegin, size_t featureValsPerPoint) {
            fstdistfunc_ = SumSquaredDist;
            //data_size_ = featureValsPerPoint * sizeof(float);
            data_size_ = sizeof(FeatureData<std::vector<int>>);

            assert((::std::sqrt(neighborhoodSize) - std::floor(::std::sqrt(neighborhoodSize))) == 0);  // neighborhoodSize must be perfect square
            unsigned int _kernelWidth = (int)::std::sqrt(neighborhoodSize);

            ::std::vector<float> A(neighborhoodSize);
            switch (weighting)
            {
            case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(A.begin(), A.end(), 1); break;
            case loc_Neigh_Weighting::WEIGHT_BINO: A = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;        // weight the center with 1
            case loc_Neigh_Weighting::WEIGHT_GAUS: A = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NONE); break;
            default:  std::fill(A.begin(), A.end(), -1);  break;  // no implemented weighting type given. 
            }

            params_ = { dataVectorBegin, dim, A, neighborhoodSize, L2Sqr };

#if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                params_.L2distfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                params_.L2distfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &params_;
        }

        ~SSDSpace() {}
    };


    // ---------------
    //    Point cloud distance (Hausdorff distances)
    // ---------------

    struct space_params_Haus {
        const float* dataVectorBegin;
        size_t dim;
        ::std::vector<float> A;         // neighborhood similarity matrix
        size_t neighborhoodSize;        //  (2 * (params._numLocNeighbors) + 1) * (2 * (params._numLocNeighbors) + 1)
        DISTFUNC<float> L2distfunc_;
    };


    static float
        HausdorffDist(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        // TODO: switch from IDs to actual values, maybe?
        std::vector<int> idsN1 = static_cast<FeatureData<std::vector<int>>*>((IFeatureData*)pVect1v)->data; // vector with IDs in neighborhood 1
        std::vector<int> idsN2 = static_cast<FeatureData<std::vector<int>>*>((IFeatureData*)pVect2v)->data;

        // parameters
        space_params_Haus* sparam = (space_params_Haus*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        const float* dataVectorBegin = sparam->dataVectorBegin;
        const std::vector<float> weights = sparam->A;
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        std::vector<float> colDistMins(neighborhoodSize, FLT_MAX);
        std::vector<float> rowDistMins(neighborhoodSize, FLT_MAX);
        float distN1N2 = 0;

        float maxN1 = 0;
        float maxN2 = 0;

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2 (colDist) and vice versa (rowDist)
        // Weight the colDist and rowDist with the inverse of the number of items in the neighborhood
        for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {

            for (size_t n2 = 0; n2 < neighborhoodSize; n2++) {

                distN1N2 = L2distfunc_(dataVectorBegin + (idsN1[n1] * ndim), dataVectorBegin + (idsN2[n2] * ndim), &ndim);

                if (distN1N2 < colDistMins[n1])
                    colDistMins[n1] = distN1N2;

                if (distN1N2 < rowDistMins[n2])
                    rowDistMins[n2] = distN1N2;
            }
        }

        // find largest of mins
        for (size_t n = 0; n < neighborhoodSize; n++) {
            if (weights[n] * colDistMins[n] > maxN1)
                maxN1 = weights[n] * colDistMins[n];

            if (weights[n] * rowDistMins[n] > maxN2)
                maxN2 = weights[n] * rowDistMins[n];
        }

        assert(maxN1 < FLT_MAX);
        assert(maxN2 < FLT_MAX);

        return std::max(maxN1, maxN2);
    }


    class HausdorffSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Haus params_;

    public:
        HausdorffSpace(size_t dim, size_t neighborhoodSize, loc_Neigh_Weighting weighting, const float* dataVectorBegin, size_t featureValsPerPoint) {
            fstdistfunc_ = HausdorffDist;
            //data_size_ = featureValsPerPoint * sizeof(float);
            data_size_ = sizeof(FeatureData<std::vector<float>>);

            assert((::std::sqrt(neighborhoodSize) - std::floor(::std::sqrt(neighborhoodSize))) == 0);  // neighborhoodSize must be perfect square
            unsigned int _kernelWidth = (int)::std::sqrt(neighborhoodSize);

            ::std::vector<float> A(neighborhoodSize);
            switch (weighting)
            {
            case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(A.begin(), A.end(), 1); break;
            case loc_Neigh_Weighting::WEIGHT_BINO: A = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;        // weight the center with 1
            case loc_Neigh_Weighting::WEIGHT_GAUS: A = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NONE); break;
            default:  std::fill(A.begin(), A.end(), -1);  break;  // no implemented weighting type given. 
            }

            params_ = { dataVectorBegin, dim, A, neighborhoodSize, L2Sqr };

#if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                params_.L2distfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                params_.L2distfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &params_;
        }

        ~HausdorffSpace() {}
    };

    // ---------------
    //    Point cloud distance (Hausdorff _median distances)
    // ---------------


    static float
        HausdorffDist_median(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        // TODO: switch from IDs to actual values, maybe?
        std::vector<int> idsN1 = static_cast<FeatureData<std::vector<int>>*>((IFeatureData*)pVect1v)->data; // vector with IDs in neighborhood 1
        std::vector<int> idsN2 = static_cast<FeatureData<std::vector<int>>*>((IFeatureData*)pVect2v)->data;

        // parameters
        space_params_Haus* sparam = (space_params_Haus*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        const float* dataVectorBegin = sparam->dataVectorBegin;
        const std::vector<float> weights = sparam->A;
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        std::vector<float> colDistMins(neighborhoodSize, FLT_MAX);
        std::vector<float> rowDistMins(neighborhoodSize, FLT_MAX);
        float distN1N2 = 0;

        float colMed = FLT_MAX;
        float rowMed = FLT_MAX;

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2 (colDist) and vice versa (rowDist)
        // Weight the colDist and rowDist with the inverse of the number of items in the neighborhood
        for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {

            for (size_t n2 = 0; n2 < neighborhoodSize; n2++) {

                distN1N2 = L2distfunc_(dataVectorBegin + (idsN1[n1] * ndim), dataVectorBegin + (idsN2[n2] * ndim), &ndim);

                if (distN1N2 < colDistMins[n1])
                    colDistMins[n1] = distN1N2;

                if (distN1N2 < rowDistMins[n2])
                    rowDistMins[n2] = distN1N2;
            }
        }

		assert(neighborhoodSize % 2 != 0); 

		// weight minima
		std::transform(colDistMins.begin(), colDistMins.end(), weights.begin(), colDistMins.begin(), [](float min, float weight) {if (min < FLT_MAX) { return min * weight; } else { return FLT_MAX; }});
		std::transform(rowDistMins.begin(), rowDistMins.end(), weights.begin(), rowDistMins.begin(), [](float min, float weight) {if (min < FLT_MAX) { return min * weight; } else { return FLT_MAX; }});

		// count FLT_MAX to determine median pos
		size_t neighborhoodSize_c = neighborhoodSize - std::count(colDistMins.begin(), colDistMins.end(), FLT_MAX);
		size_t neighborhoodSize_r = neighborhoodSize - std::count(rowDistMins.begin(), rowDistMins.end(), FLT_MAX);

		// find median of mins
		colMed = CalcMedian(colDistMins, neighborhoodSize_c);
		rowMed = CalcMedian(rowDistMins, neighborhoodSize_r);

        assert(colMed < FLT_MAX);
        assert(rowMed < FLT_MAX);

        return (colMed + rowMed) / 2;
    }

    class HausdorffSpace_median : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Haus params_;

    public:
        HausdorffSpace_median(size_t dim, size_t neighborhoodSize, loc_Neigh_Weighting weighting, const float* dataVectorBegin, size_t featureValsPerPoint) {
            fstdistfunc_ = HausdorffDist_median;
            //data_size_ = featureValsPerPoint * sizeof(float);
            data_size_ = sizeof(FeatureData<std::vector<int>>);

            assert((::std::sqrt(neighborhoodSize) - std::floor(::std::sqrt(neighborhoodSize))) == 0);  // neighborhoodSize must be perfect square
            unsigned int _kernelWidth = (int)::std::sqrt(neighborhoodSize);

            ::std::vector<float> A(neighborhoodSize);
            switch (weighting)
            {
            case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(A.begin(), A.end(), 1); break;
            case loc_Neigh_Weighting::WEIGHT_BINO: A = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;        // weight the center with 1
            case loc_Neigh_Weighting::WEIGHT_GAUS: A = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NONE); break;
            default:  std::fill(A.begin(), A.end(), -1);  break;  // no implemented weighting type given. 
            }

            params_ = { dataVectorBegin, dim, A, neighborhoodSize, L2Sqr };

#if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                params_.L2distfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                params_.L2distfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &params_;
        }

        ~HausdorffSpace_median() {}
    };


    // ---------------
    //    Bhattacharyya Space
    // ---------------

    struct space_params_Bhattacharyya {

    };

    inline float distBhattacharyya(const Eigen::VectorXf& mean1, const Eigen::MatrixXf& covmat1, const float det1, const Eigen::VectorXf& mean2, const Eigen::MatrixXf& covmat2, const float det2) {
    	Eigen::MatrixXf covmat_comb = (covmat1 + covmat2) / 2.0f;
    	Eigen::VectorXf mean_diff = mean1 - mean2;
    	return 0.125f * mean_diff.transpose() * covmat_comb.inverse() * mean_diff + 0.5f * std::logf(covmat_comb.determinant() / std::sqrt(det1 * det2));
    }


    static float
        Bhattacharyya(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        // pointer to the data features, not the actual data
        FeatureData<multivar_normal_plusDet>* pVect1 = static_cast<FeatureData<multivar_normal_plusDet>*>((IFeatureData*)pVect1v);
        FeatureData<multivar_normal_plusDet>* pVect2 = static_cast<FeatureData<multivar_normal_plusDet>*>((IFeatureData*)pVect2v);
        
        // Bhattacharyya distance
        return distBhattacharyya(std::get<0>(pVect1->data), std::get<1>(pVect1->data), std::get<2>(pVect1->data), std::get<0>(pVect2->data), std::get<1>(pVect2->data), std::get<2>(pVect2->data));
    }

    class Bhattacharyya_Space : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        space_params_Bhattacharyya params_;

    public:
        Bhattacharyya_Space() {

            fstdistfunc_ = Bhattacharyya;

            data_size_ = sizeof(FeatureData<multivar_normal_plusDet>);

            params_ = { };

        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return (void *)&params_;
        }

        ~Bhattacharyya_Space() {}
    };

}
