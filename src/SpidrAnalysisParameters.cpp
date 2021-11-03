#include "SpidrAnalysisParameters.h"


namespace logging {

    std::string distance_metric_name(const distance_metric& metric) {
        switch (metric) {
        case distance_metric::METRIC_QF: return "Quadratic form distance";
        case distance_metric::METRIC_HEL: return "Hellinger distance";
        case distance_metric::METRIC_EUC: return "Euclidean distance (squared)";
        case distance_metric::METRIC_CHA: return "Chamfer distance (point cloud)";
        case distance_metric::METRIC_SSD: return "Sum of squared distances (point cloud)";
        case distance_metric::METRIC_HAU: return "Hausdorff distance (point cloud)";
        case distance_metric::METRIC_HAU_med: return "Hausdorff distance (point cloud) but with median instead of max";
        case distance_metric::METRIC_BHATTACHARYYA: return "Bhattacharyya distance between two multivariate normal distributions";
        case distance_metric::METRIC_DETMATRATIO: return "Deteterminant Ratio part of Bhattacharyya distance";
        case distance_metric::METRIC_CMD_covmat: return "Correlation Matrix distance";
        case distance_metric::METRIC_FRECHET_Gen: return "Fr�chet distance";
        case distance_metric::METRIC_FRECHET_CovMat: return "Fr�chet distance but ignoring the means";
        case distance_metric::METRIC_FROBENIUS_CovMat: return "Frobenius norm of element-wise differences";
        default: return "";
        }
    }

    std::string neighborhood_weighting_name(const loc_Neigh_Weighting& weighting) {
        switch (weighting) {
        case loc_Neigh_Weighting::WEIGHT_UNIF: return "Uniform weighting";
        case loc_Neigh_Weighting::WEIGHT_BINO: return "Binomial weighting";
        case loc_Neigh_Weighting::WEIGHT_GAUS: return "Gaussian weighting";
        default: return "";
        }
    }

}


std::tuple< feature_type, distance_metric> get_feat_and_dist(feat_dist feat_dist)
{
    feature_type feat;
    distance_metric dist;

    switch (feat_dist) {
    case feat_dist::HIST_QF:
        feat = feature_type::TEXTURE_HIST_1D;
        dist = distance_metric::METRIC_QF;
        break;
    case feat_dist::HIST_HEL:
        feat = feature_type::TEXTURE_HIST_1D;
        dist = distance_metric::METRIC_HEL;
        break;
    case feat_dist::LMI_EUC:
        feat = feature_type::LOCALMORANSI;
        dist = distance_metric::METRIC_EUC;
        break;
    case feat_dist::LGC_EUC:
        feat = feature_type::LOCALGEARYC;
        dist = distance_metric::METRIC_EUC;
        break;
    case feat_dist::PC_CHA:
        feat = feature_type::PCLOUD;
        dist = distance_metric::METRIC_CHA;
        break;
    case feat_dist::PC_HAU:
        feat = feature_type::PCLOUD;
        dist = distance_metric::METRIC_HAU;
        break;
    case feat_dist::MVN_BHAT:
        feat = feature_type::MULTIVAR_NORM;
        dist = distance_metric::METRIC_BHATTACHARYYA;
        break;
    case feat_dist::MVN_FRO:
        feat = feature_type::MULTIVAR_NORM;
        dist = distance_metric::METRIC_FROBENIUS_CovMat;
        break;
    case feat_dist::CHIST_EUC:
        feat = feature_type::CHANNEL_HIST;
        dist = distance_metric::METRIC_EUC;
        break;
    }

    return { feat , dist };
}