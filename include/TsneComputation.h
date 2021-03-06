#pragma once

#ifdef __APPLE__
#include "glad/glad_3_3.h"
#define __gl3_h_
#endif
#include <GLFW/glfw3.h>

#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/dimensionality_reduction/sparse_tsne_user_def_probabilities.h"
#include "hdi/dimensionality_reduction/gradient_descent_tsne_texture.h"

#include <vector>
#include <string>

class SpidrParameters;

class TsneComputation
{
public:
    TsneComputation();

    void setVerbose(bool verbose);
    void setIterations(int iterations);
    void setExaggerationIter(int exaggerationIter);
    void setExponentialDecay(int exponentialDecay);
    void setPerplexity(int perplexity);
    void setNumDimensionsOutput(int numDimensionsOutput);

    inline bool verbose() { return _verbose; }
    inline int iterations() { return _iterations; }
    inline int exaggerationIter() { return _exaggerationIter; }
    inline int perplexity() { return _perplexity; }
    inline int numDimensionsOutput() { return _numDimensionsOutput; }

    /*!
     * 
     * 
     * \param knn_indices
     * \param knn_distances
     * \param params
     */
    void setup(const std::vector<int> knn_indices, const std::vector<float> knn_distances, const SpidrParameters params);
    
    /*!
     * 
     * 
     */
    void initTSNE();
    void stopGradientDescent();
    void markForDeletion();

    /*!
     * !
     * 
     */
    void compute();

    /*!
     * 
     * 
     * \return 
     */
    const std::vector<float>& output();

    inline bool isTsneRunning() { return _isTsneRunning; }
    inline bool isGradientDescentRunning() { return _isGradientDescentRunning; }
    inline bool isMarkedForDeletion() { return _isMarkedForDeletion; }

private:
    void computeGradientDescent();
    void initGradientDescent();
    void embed();
    void copyFloatOutput();

private:
    // TSNE structures
    hdi::dr::HDJointProbabilityGenerator<float>::sparse_scalar_matrix_type _probabilityDistribution;
    hdi::dr::SparseTSNEUserDefProbabilities<float> _A_tSNE;
    hdi::dr::GradientDescentTSNETexture _GPGPU_tSNE;
    hdi::data::Embedding<float> _embedding;

    // Data
    std::vector<int> _knn_indices;               /*!<> */
    std::vector<float> _knn_distances;           /*!<> */
    size_t _numPoints;                            /*!<> */
    std::vector<float> _outputData;                     /*!<> */

    // Options
    int _iterations;                                    /*!<> */
    int _numTrees;
    int _numChecks;
    int _exaggerationIter;
    int _exponentialDecay;
    int _perplexity;                                    /*!<> */
    int _perplexity_multiplier;
    int _numDimensionsOutput;
    int _nn;                                            /*!<> */

    // Evaluation (for determining the filename when saving the embedding to disk)
    std::string _embeddingName;                     /*!< Name of the embedding */
    size_t _numDataDims;

    // Flags
    bool _verbose;
    bool _isGradientDescentRunning;
    bool _isTsneRunning;
    bool _isMarkedForDeletion;

    int _continueFromIteration;
	GLFWwindow* _offscreen_context;
};
