#ifndef ST_HANDEYE_GRAPH_HPP
#define ST_HANDEYE_GRAPH_HPP

#include <vector>
#include <Eigen/Dense>

namespace st_handeye {

/**
 * @brief Optimization parameters
 */
struct OptimizationParams {
public:
    OptimizationParams()
        : world2hand_inf_scale_trans(0.01),
          world2hand_inf_scale_rot(1.0),
          pattern2d_inf_scale(1.0),
          source_inf_scale(1.0),
          solver_name("lm_var_cholmod"),
          num_iterations(8192),
          robust_kernel_handpose("Huber"),
          robust_kernel_projection("Huber"),
          robust_kernel_source("Huber"),
          robust_kernel_handpose_delta(0.01),
          robust_kernel_projection_delta(1.0),
          robust_kernel_source_delta(1.0)
    {}

    // scale of information matrices
    double world2hand_inf_scale_trans;      // translation part of world2hand transformation
    double world2hand_inf_scale_rot;        // rotation part of world2hand transformation
    double pattern2d_inf_scale;             // projection
    double source_inf_scale;                // source point (source-detector model only)

    // optimization settings
    std::string solver_name;                // gn_var, lm_var, gn_var_cholmod, lm_var_cholmod ...
    int num_iterations;                     // maximum number of iterations

    // robust kernels
    std::string robust_kernel_handpose;     // robust kernel applied to handpose edges
    std::string robust_kernel_projection;   // robust kernel applied to projection edges
    std::string robust_kernel_source;       // robust kernel applied to source edges (source-detector model only)
    double robust_kernel_handpose_delta;    // robust kernel delta
    double robust_kernel_projection_delta;  // 
    double robust_kernel_source_delta;      // (source-detector model only)
};

/**
 * @brief estimate object2eye transformations from 2D points using solvePnP
 * @param camera_matrix
 * @param pattern_3d
 * @param pattern_2ds
 * @return estimated object2eye transformations
 */
std::vector<Eigen::Isometry3d> calc_object2eyes(
        const Eigen::Matrix3d& camera_matrix,
        const Eigen::MatrixXd& pattern_3d,
        const std::vector<Eigen::MatrixXd>& pattern_2ds
);

/**
 * @brief reprojection error minimization-based hand-eye calibration
 * @param camera_matrix    camera intrinsic parameters
 * @param pattern_3d       3d coordinates of the points on the calibration pattern
 * @param world2hands      world2hand transformations (hand poses[N])
 * @param pattern_2ds      visually detected points of the pattern (Matrix<M, 3> * N)
 * @param hand2eye         (input) initial guess -> (output) estimated hand2eye transformation
 * @param object2world     (input) initial guess -> (output) estimated object2world transformation
 * @param params           optimization parameters
 * @return if calibration successed
 *
 * @note We recommend to first use spatial_calibration_visp to obtain an initial guess, 
 *       and then the input the result to this function
 */
bool spatial_calibration_graph (
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::MatrixXd& pattern_3d,
    const std::vector<Eigen::Isometry3d>& world2hands,
    const std::vector<Eigen::MatrixXd>& pattern_2ds,
    Eigen::Isometry3d& hand2eye,
    Eigen::Isometry3d& object2world,
    OptimizationParams params = OptimizationParams()
);

/**
 * @brief hand-eye calibration using Tsai's algorithm in VISP
 * @param camera_matrix    camera intrinsic parameters
 * @param pattern_3d       3d coordinates of the points on the calibration pattern
 * @param world2hands      world2hand transformations (hand poses[N])
 * @param pattern_2ds      visually detected points of the pattern (Matrix<M, 3> * N)
 * @param hand2eye         (input) initial guess -> (output) estimated hand2eye transformation
 * @param object2world     (input) initial guess -> (output) estimated object2world transformation
 * @param params           optimization parameters (not used in this function)
 * @return if calibration successed
 */
bool spatial_calibration_visp (
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::MatrixXd& pattern_3d,
    const std::vector<Eigen::Isometry3d>& world2hands,
    const std::vector<Eigen::MatrixXd>& pattern_2ds,
    Eigen::Isometry3d& hand2eye,
    Eigen::Isometry3d& object2world,
    OptimizationParams params = OptimizationParams()
);

/**
 * @brief hand-eye calibration using Dual Quaternions-based method (handeye_calib_camodocal)
 * @param camera_matrix    camera intrinsic parameters
 * @param pattern_3d       3d coordinates of the points on the calibration pattern
 * @param world2hands      world2hand transformations (hand poses[N])
 * @param pattern_2ds      visually detected points of the pattern (Matrix<M, 3> * N)
 * @param hand2eye         (input) initial guess -> (output) estimated hand2eye transformation
 * @param object2world     (input) initial guess -> (output) estimated object2world transformation
 * @param params           optimization parameters (not used in this function)
 * @return if calibration successed
 */
bool spatial_calibration_dualquaternion (
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::MatrixXd& pattern_3d,
    const std::vector<Eigen::Isometry3d>& world2hands,
    const std::vector<Eigen::MatrixXd>& pattern_2ds,
    Eigen::Isometry3d& hand2eye,
    Eigen::Isometry3d& object2world,
    OptimizationParams params = OptimizationParams()
);

/**
 * @brief hand-eye calibration with a source-detector projection model
 * @param camera_matrix            detector intrinsic parameters
 * @param pattern_3d               3d coordinates of the points on the calibration pattern
 * @param world2hands_detector     world2detector transformations
 * @param world2hands_source       world2source transformations
 * @param pattern_2ds              visually observed pattern points
 * @param hand2detector            (input) initial guess and (output) estimated hand2detector transformation
 * @param hand2source              (input) initial guess and (output) estimated hand2source transformation
 * @param object2world             (input) initial guess and (output) estimated object2world transformation
 * @param params                   optimization parameters
 * @return if the calibration successed
 */
bool spatial_calibration_graph_xray (
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::MatrixXd& pattern_3d,
    const std::vector<Eigen::Isometry3d>& world2hands_detector,
    const std::vector<Eigen::Isometry3d>& world2hands_source,
    const std::vector<Eigen::MatrixXd>& pattern_2ds,
    Eigen::Isometry3d& hand2detector,
    Eigen::Vector3d& hand2source,
    Eigen::Isometry3d& object2world,
    OptimizationParams params = OptimizationParams()
);


}

#endif
