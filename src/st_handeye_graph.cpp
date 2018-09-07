#include <iostream>
#include <g2o/stuff/macros.h>
#include <g2o/core/factory.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>

#include <g2o/edge_projection_pinhole.hpp>
#include <g2o/edge_handeye_transform.hpp>
#include <g2o/parameter_point_xyz.hpp>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

#include <st_handeye/st_handeye.hpp>

G2O_USE_OPTIMIZATION_LIBRARY(csparse)
G2O_USE_OPTIMIZATION_LIBRARY(cholmod)

namespace st_handeye {

/**
 * @brief Reprojection error minimization-based hand-eye calibration with a pinhole camera
 */
class SpatialCalibration {
public:
    SpatialCalibration() {}
    ~SpatialCalibration() {}

    bool calibrate(const Eigen::Matrix3d& camera_matrix, const Eigen::MatrixXd& pattern_3d, const std::vector<Eigen::Isometry3d>& world2hands, const std::vector<Eigen::MatrixXd>& pattern_2ds, Eigen::Isometry3d& hand2eye, Eigen::Isometry3d& object2world, const OptimizationParams& params) {
        graph.reset(initialize_graph(params.solver_name));
        if(!graph) {
            std::cerr << "failed to construct pose graph" << std::endl;
            return false;
        }
        g2o::RobustKernelFactory* kernel_factory = g2o::RobustKernelFactory::instance();

        if(!install_parameters(camera_matrix, pattern_3d)) {
            std::cerr << "failed to install parameters" << std::endl;
            return false;
        }

        hand2eye_vertex = new g2o::VertexSE3();
        hand2eye_vertex->setEstimate(hand2eye);
        hand2eye_vertex->setId(graph->vertices().size());
        graph->addVertex(hand2eye_vertex);

        object2world_vertex = new g2o::VertexSE3();
        object2world_vertex->setEstimate(object2world);
        object2world_vertex->setId(graph->vertices().size());
        graph->addVertex(object2world_vertex);

        for(int i=0; i<world2hands.size(); i++) {
            // object2eye vertex
            Eigen::Isometry3d object2eye = hand2eye * world2hands[i] * object2world;
            g2o::VertexSE3* object2eye_vertex = new g2o::VertexSE3();
            object2eye_vertex->setEstimate(object2eye);
            object2eye_vertex->setId(graph->vertices().size());
            graph->addVertex(object2eye_vertex);
            object2eye_vertices.push_back(object2eye_vertex);

            // handeye transform edge
            g2o::EdgeHandeyeTransform* handeye_trans_edge = new g2o::EdgeHandeyeTransform();
            handeye_trans_edge->vertices()[0] = hand2eye_vertex;
            handeye_trans_edge->vertices()[1] = object2world_vertex;
            handeye_trans_edge->vertices()[2] = object2eye_vertex;
            handeye_trans_edge->setMeasurement(world2hands[i]);

            Eigen::MatrixXd handeye_inf = Eigen::MatrixXd::Identity(6, 6);
            handeye_inf.topLeftCorner(3, 3) *= params.world2hand_inf_scale_trans;
            handeye_inf.bottomRightCorner(3, 3) *= params.world2hand_inf_scale_rot;

            handeye_trans_edge->setInformation(handeye_inf);
            handeye_trans_edge->setId(graph->edges().size());
            graph->addEdge(handeye_trans_edge);
            handeye_trans_edges.push_back(handeye_trans_edge);

            if(params.robust_kernel_handpose != "NONE") {
                g2o::RobustKernel* kernel = kernel_factory->construct(params.robust_kernel_handpose);
                if(kernel == nullptr) {
                    std::cerr << "invalid kernel name!!" << std::endl;
                    return false;
                }
                kernel->setDelta(params.robust_kernel_handpose_delta);
                handeye_trans_edge->setRobustKernel(kernel);
            }

            // projection edge
            const auto& pattern_2d = pattern_2ds[i];
            for(int j=0; j<pattern_2d.cols(); j++) {
                g2o::EdgeProjectionPinhole* projection_edge = new g2o::EdgeProjectionPinhole();
                projection_edge->setMeasurement(pattern_2d.col(j));
                projection_edge->setInformation(Eigen::Matrix2d::Identity() * params.pattern2d_inf_scale);
                projection_edge->vertices()[0] = object2eye_vertex;
                projection_edge->setParameterId(0, 0);
                projection_edge->setParameterId(1, 1 + j);

                projection_edge->setId(graph->edges().size());
                graph->addEdge(projection_edge);

                if(params.robust_kernel_projection != "NONE") {
                    g2o::RobustKernel* kernel = kernel_factory->construct(params.robust_kernel_projection);
                    if(kernel == nullptr) {
                        std::cerr << "invalid kernel name!!" << std::endl;
                        return false;
                    }
                    kernel->setDelta(params.robust_kernel_projection_delta);
                    projection_edge->setRobustKernel(kernel);
                }

            }
        }

        std::cout << "optimizing..." << std::endl;

        int max_iterations = params.num_iterations;
        graph->initializeOptimization();
        graph->setVerbose(false);

        double chi2 = graph->chi2();
        int iterations = graph->optimize(max_iterations);

        std::cout << "iterations: " << iterations << "/" << max_iterations << std::endl;
        std::cout << "chi2: (before)" << chi2 << " -> (after)" << graph->chi2() << std::endl;

        hand2eye = hand2eye_vertex->estimate();
        object2world = object2world_vertex->estimate();

        return true;
    }

private:
    g2o::SparseOptimizer* initialize_graph(const std::string& solver_name = "lm_var_cholmod") {
        g2o::SparseOptimizer* graph = new g2o::SparseOptimizer();

        std::cout << "construct solver... " << std::flush;
        g2o::OptimizationAlgorithmFactory* solver_factory = g2o::OptimizationAlgorithmFactory::instance();
        g2o::OptimizationAlgorithmProperty solver_property;
        g2o::OptimizationAlgorithm* solver = solver_factory->construct(solver_name, solver_property);
        graph->setAlgorithm(solver);

        if (!graph->solver()) {
            std::cerr << std::endl;
            std::cerr << "error : failed to allocate solver!!" << std::endl;
            solver_factory->listSolvers(std::cerr);
            std::cerr << "-------------" << std::endl;
            std::cin.ignore(1);
            delete graph;
            return nullptr;
        }

        std::cout << "done" << std::endl;
        return graph;
    }

    bool install_parameters(const Eigen::Matrix3d& camera_matrix, const Eigen::MatrixXd& pattern_3d) {
        std::cout << "installing parameters... " << std::flush;

        g2o::ParameterCamera* camera_param = new g2o::ParameterCamera();
        camera_param->setId(0);
        camera_param->setKcam(camera_matrix(0, 0), camera_matrix(1, 1), camera_matrix(0, 2), camera_matrix(1, 2));
        if(!graph->addParameter(camera_param)) {
            return false;
        }

        for(int i=0; i<pattern_3d.cols(); i++) {
            g2o::ParameterPointXYZ* pt_param = new g2o::ParameterPointXYZ(pattern_3d.col(i).topRows(3));
            pt_param->setId(1 + i);
            if(!graph->addParameter(pt_param)) {
                return false;
            }
        }
        std::cout << "done" << std::endl;
        return true;
    }
private:
    std::unique_ptr<g2o::SparseOptimizer> graph;

    g2o::VertexSE3* hand2eye_vertex;
    g2o::VertexSE3* object2world_vertex;
    std::vector<g2o::VertexSE3*> object2eye_vertices;

    std::vector<g2o::EdgeHandeyeTransform*> handeye_trans_edges;
};


bool spatial_calibration_graph (
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::MatrixXd& pattern_3d,
    const std::vector<Eigen::Isometry3d>& world2hands,
    const std::vector<Eigen::MatrixXd>& pattern_2ds,
    Eigen::Isometry3d& hand2eye,
    Eigen::Isometry3d& object2world,
    OptimizationParams params
) {
    SpatialCalibration calib;
    return calib.calibrate(camera_matrix, pattern_3d, world2hands, pattern_2ds, hand2eye, object2world, params);
}

}
