#include "SpidrAnalysisParameters.h"


namespace logging {

    std::string distance_metric_name(const distance_metric& metric) {
        switch (metric) {
        case distance_metric::METRIC_QF: return "Quadratic form distance";
        case distance_metric::METRIC_HEL: return "Hellinger distance";
        case distance_metric::METRIC_EUC: return "Euclidean distance";
        case distance_metric::METRIC_CHA: return "Chamfer distance (point collection)";
        case distance_metric::METRIC_SSD: return "Sum of squared distances (point cloud)";
        case distance_metric::METRIC_HAU: return "Hausdorff distance (point cloud)";
        case distance_metric::METRIC_HAU_med: return "Hausdorff distance (point cloud) but with median instead of max";
        case distance_metric::METRIC_BHATTACHARYYA: return "Bhattacharyya distance between two multivariate normal distributions";
        case distance_metric::METRIC_DETMATRATIO: return "Deteterminant Ratio part of Bhattacharyya distance";
        case distance_metric::METRIC_CMD_covmat: return "Correlation Matrix distance";
        case distance_metric::METRIC_FRECHET_Gen: return "Fréchet distance";
        case distance_metric::METRIC_FRECHET_CovMat: return "Fréchet distance but ignoring the means";
        case distance_metric::METRIC_FROBENIUS_CovMat: return "Frobenius norm of element-wise differences";
        default: return "";
        }
    }
}