#include <memory>
#include <random>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

#include <st_handeye/st_handeye.hpp>

class HandeyeSimulation {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    HandeyeSimulation(const Eigen::Matrix3d& camera_matrix, const Eigen::MatrixXd& pattern_3d, const Eigen::Isometry3d& hand2eye, const Eigen::Isometry3d& object2world)
        : camera_matrix(camera_matrix),
          pattern_3d(pattern_3d),
          hand2eye(hand2eye),
          object2world(object2world)
    {}

    Eigen::MatrixXd project(const Eigen::Isometry3d& world2hand) const {
        Eigen::Isometry3d object2eye = hand2eye * world2hand * object2world;

        Eigen::MatrixXd transformed = object2eye.matrix() * pattern_3d;
        Eigen::MatrixXd uvs = camera_matrix * transformed.topRows(3);

        uvs.row(0) = uvs.row(0).array() / uvs.row(2).array();
        uvs.row(1) = uvs.row(1).array() / uvs.row(2).array();

        return uvs.topRows(2);
    }

    void visualize(const Eigen::MatrixXd& pattern_2d, int sleep=100) const {
        cv::Mat canvas(480, 640, CV_8UC3, cv::Scalar::all(0));
        for(int i=0; i<pattern_2d.cols(); i++) {
            cv::Point pt(pattern_2d(0, i), pattern_2d(1, i));
            cv::circle(canvas, pt, 5, cv::Scalar(255, 0, 0), -1);
        }

        cv::imshow("canvas", canvas);
        cv::waitKey(sleep);
    }

private:
    Eigen::Matrix3d camera_matrix;
    Eigen::MatrixXd pattern_3d;

    Eigen::Isometry3d hand2eye;
    Eigen::Isometry3d object2world;
};

int main(int argc, char** argv) {
    using namespace boost::program_options;

    options_description description("handeye_simulation");
    description.add_options()
            ("seed,s", value<long>()->default_value(0), "seed of random")
            ("visualize,v", value<bool>()->default_value(false), "if visualize")
            ("x_steps", value<int>()->default_value(1), "x steps")
            ("y_steps", value<int>()->default_value(1), "y steps")
            ("z_steps", value<int>()->default_value(1), "z steps")
            ("x_step", value<double>()->default_value(0.5), "x step")
            ("y_step", value<double>()->default_value(0.5), "y step")
            ("z_step", value<double>()->default_value(0.25), "z step")
            ("z_offset", value<double>()->default_value(0.5), "z offset")
            ("hand2eye_trans", value<double>()->default_value(0.1), "stddev of the translation of the handeye transformation")
            ("hand2eye_rot", value<double>()->default_value(10.0), "stddev of the rotation angle [deg] of the handeye transformation")
            ("tnoise", value<double>()->default_value(0.0), "stddev of the translation noise of the world2hand transformation")
            ("rnoise", value<double>()->default_value(0.0), "stddev of the rotation noise [deg] of the world2hand transformation")
            ("vnoise", value<double>()->default_value(0.0), "stddev of the visual noise on the marker detection")
            ("visual_inf_scale", value<double>()->default_value(1.0), "scale of the informatoin matrix for visual detections")
            ("handpose_inf_scale_trans", value<double>()->default_value(1.0), "scale of the informatoin matrix for hand poses")
            ("handpose_inf_scale_rot", value<double>()->default_value(1.0), "scale of the informatoin matrix for hand poses")
            ("num_iterations", value<int>()->default_value(8192), "max number of iterations")
            ("solver_name", value<std::string>()->default_value("lm_var_cholmod"), "g2o solver name")
            ("robust_kernel_handpose", value<std::string>()->default_value("NONE"), "robust kernel for handpose edges")
            ("robust_kernel_projection", value<std::string>()->default_value("NONE"), "robust kernel for projection edges")
            ("robust_kernel_handpose_delta", value<double>()->default_value(1.0), "robust kernel delta for handpose edges")
            ("robust_kernel_projection_delta", value<double>()->default_value(1.0), "robust kernel delta for projection edges")
            ("use_init_guess", value<bool>()->default_value(false), "if use tsai's result as an initial guess")
    ;

    variables_map vm;
    store(parse_command_line(argc, argv, description), vm);
    notify(vm);

    std::mt19937 mt(vm["seed"].as<long>());
    srand(mt());

    Eigen::Matrix3d camera_matrix = Eigen::Matrix3d::Identity();
    camera_matrix(0, 0) = 300.0;
    camera_matrix(1, 1) = 300.0;
    camera_matrix(0, 2) = 320.0;
    camera_matrix(1, 2) = 240.0;

    int rows = 7;
    int cols = 5;
    double dimension = 0.1;
    Eigen::MatrixXd pattern_3d(4, rows * cols);
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            pattern_3d.col(i * cols + j) = Eigen::Vector4d(i - (rows - 1) / 2.0, j - (cols - 1)/2.0, 0, 1);
            pattern_3d.col(i * cols + j).topRows(3) *= dimension;
        }
    }

    Eigen::Isometry3d object2world = Eigen::Isometry3d::Identity();
    object2world.translation() = Eigen::Vector3d(1.0, 0.0, 0);

    double hand2eye_trans = std::normal_distribution<>(0.0, vm["hand2eye_trans"].as<double>())(mt);
    double hand2eye_rotangle = std::normal_distribution<>(0.0, vm["hand2eye_rot"].as<double>())(mt) * M_PI / 180.0;
    Eigen::Vector3d hand2eye_rotaxis = (Eigen::Vector3d::Random() - Eigen::Vector3d::Ones() * 0.5).normalized();

    Eigen::Isometry3d hand2eye = Eigen::Isometry3d::Identity();
    hand2eye.translation() = (Eigen::Vector3d::Random() - Eigen::Vector3d::Ones() * 0.5).normalized() * hand2eye_trans;
    hand2eye.linear() = Eigen::AngleAxisd(hand2eye_rotangle, hand2eye_rotaxis).toRotationMatrix();

    std::unique_ptr<HandeyeSimulation> sim(new HandeyeSimulation(camera_matrix, pattern_3d, hand2eye, object2world));
    pattern_3d = pattern_3d.topRows(3);

    std::vector<Eigen::Isometry3d> world2hands;
    std::vector<Eigen::MatrixXd> pattern_2ds;

    int x_steps = vm["x_steps"].as<int>();
    int y_steps = vm["y_steps"].as<int>();
    int z_steps = vm["z_steps"].as<int>();
    for(int z_step = 0; z_step <= z_steps; z_step ++) {
        for(int y_step = -y_steps; y_step <= y_steps; y_step ++) {
            for(int x_step = -x_steps; x_step <= x_steps; x_step ++) {
                Eigen::Isometry3d handpose = Eigen::Isometry3d::Identity();
                handpose.translation().x() = 1.0 + x_step * vm["x_step"].as<double>();
                handpose.translation().y() = 0.0 + y_step * vm["y_step"].as<double>();
                handpose.translation().z() = vm["z_offset"].as<double>() + z_step * vm["z_step"].as<double>();
                handpose.linear() = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()).toRotationMatrix();

                Eigen::Vector3d z_from = handpose.linear().col(2);
                Eigen::Vector3d z_to = (object2world.translation() - handpose.translation()).normalized();

                double angle = std::acos(z_from.dot(z_to));
                Eigen::Vector3d axis = z_to.cross(z_from).normalized();
                handpose.linear() = Eigen::AngleAxisd(angle, -axis).toRotationMatrix() * handpose.linear();

                Eigen::Isometry3d world2hand = hand2eye.inverse() * handpose.inverse();

                Eigen::MatrixXd pattern_2d = sim->project(world2hand);
                double visual_noise_stddev = vm["vnoise"].as<double>();
                if(visual_noise_stddev > 1e-6) {
                    for(int i=0; i<pattern_2d.cols(); i++) {
                        Eigen::Vector2d vnoise_dir = (Eigen::Vector2d::Random() - Eigen::Vector2d::Ones() * 0.5).normalized();
                        pattern_2d.col(i) += vnoise_dir * std::normal_distribution<>(0.0, visual_noise_stddev)(mt);
                    }
                }

                if(vm["visualize"].as<bool>()) {
                    sim->visualize(pattern_2d, 100);
                }

                double trans_stddev = vm["tnoise"].as<double>();
                double rot_stddev = vm["rnoise"].as<double>();
                Eigen::Vector3d transvec = (Eigen::Vector3d::Random() - Eigen::Vector3d::Ones() * 0.5).normalized();
                Eigen::Vector3d rotaxis = (Eigen::Vector3d::Random() - Eigen::Vector3d::Ones() * 0.5).normalized();

                Eigen::Isometry3d world2hand_noise = Eigen::Isometry3d::Identity();
                if(trans_stddev > 1e-6) {
                    world2hand_noise.translation() = transvec * std::normal_distribution<>(0.0, trans_stddev)(mt);
                }
                if(rot_stddev > 1e-6) {
                    double angle = std::normal_distribution<>(0.0, rot_stddev)(mt) * M_PI / 180.0;
                    world2hand_noise.linear() = Eigen::AngleAxisd(angle, rotaxis).toRotationMatrix();
                }
                world2hand = world2hand_noise * world2hand;

                world2hands.push_back(world2hand);
                pattern_2ds.push_back(pattern_2d);
            }
        }
    }

    std::cout << "num_images: " << world2hands.size() << std::endl;
    std::cout << "calibrating..." << std::endl;

    st_handeye::OptimizationParams params;
    params.pattern2d_inf_scale = vm["visual_inf_scale"].as<double>();
    params.world2hand_inf_scale_trans = vm["handpose_inf_scale_trans"].as<double>();
    params.world2hand_inf_scale_rot = vm["handpose_inf_scale_rot"].as<double>();
    params.num_iterations = vm["num_iterations"].as<int>();
    params.solver_name = vm["solver_name"].as<std::string>();
    params.robust_kernel_handpose = vm["robust_kernel_handpose"].as<std::string>();
    params.robust_kernel_projection = vm["robust_kernel_projection"].as<std::string>();
    params.robust_kernel_handpose_delta = vm["robust_kernel_handpose_delta"].as<double>();
    params.robust_kernel_projection_delta = vm["robust_kernel_projection_delta"].as<double>();

    Eigen::Isometry3d hand2eye_visp = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d object2world_visp = Eigen::Isometry3d::Identity();
    if(!st_handeye::spatial_calibration_visp(camera_matrix, pattern_3d, world2hands, pattern_2ds, hand2eye_visp, object2world_visp, params)) {
        std::cout << "failed to calibrate..." << std::endl;
        return 0;
    }

    Eigen::Isometry3d hand2eye_dq = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d object2world_dq = Eigen::Isometry3d::Identity();
    if(!st_handeye::spatial_calibration_dualquaternion(camera_matrix, pattern_3d, world2hands, pattern_2ds, hand2eye_dq, object2world_dq, params)) {
        std::cout << "failed to calibrate..." << std::endl;
        // return 0;
    }

    // Eigen::Isometry3d hand2eye_graph = hand2eye_visp;
    Eigen::Isometry3d hand2eye_graph = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d object2world_graph = Eigen::Isometry3d::Identity();
    if(vm["use_init_guess"].as<bool>()) {
        hand2eye_graph = hand2eye_visp;
        object2world_graph = object2world_visp;
    }

    if(!st_handeye::spatial_calibration_graph(camera_matrix, pattern_3d, world2hands, pattern_2ds, hand2eye_graph, object2world_graph, params)) {
        std::cout << "failed to calibrate..." << std::endl;
        return 0;
    }

    std::cout << "*** calibration results ***" << std::endl;
    std::cout << "--- hand2eye ---\n" << hand2eye.matrix() << std::endl;
    std::cout << "--- hand2eye_graph ---\n" << hand2eye_graph.matrix() << std::endl;
    std::cout << "--- hand2eye_visp ---\n" << hand2eye_visp.matrix() << std::endl;
    std::cout << "--- hand2eye_dq ---\n" << hand2eye_dq.matrix() << std::endl;

    std::cout << "--- object2world ---\n" << object2world.matrix() << std::endl;
    std::cout << "--- object2world_graph ---\n" << object2world_graph.matrix() << std::endl;
    std::cout << "--- object2world_visp ---\n" << object2world_visp.matrix() << std::endl;
    std::cout << "--- object2world_dq ---\n" << object2world_dq.matrix() << std::endl;

    double rad2deg = 180.0 / M_PI;

    Eigen::Isometry3d delta_graph = hand2eye.inverse() * hand2eye_graph;
    double terror_graph = delta_graph.translation().norm();
    double rerror_graph = Eigen::AngleAxisd().fromRotationMatrix(delta_graph.linear()).angle() * rad2deg;

    Eigen::Isometry3d delta_visp = hand2eye.inverse() * hand2eye_visp;
    double terror_visp = delta_visp.translation().norm();
    double rerror_visp = Eigen::AngleAxisd().fromRotationMatrix(delta_visp.linear()).angle() * rad2deg;

    Eigen::Isometry3d delta_dq = hand2eye.inverse() * hand2eye_dq;
    double terror_dq = delta_dq.translation().norm();
    double rerror_dq = Eigen::AngleAxisd().fromRotationMatrix(delta_dq.linear()).angle() * rad2deg;


    std::cout << "terror_graph " << terror_graph << " rerror_graph " << rerror_graph << " terror_visp " << terror_visp << " rerror_visp " << rerror_visp << " terror_dq " << terror_dq << " rerror_dq " << rerror_dq << std::endl;

	return 0;
}
