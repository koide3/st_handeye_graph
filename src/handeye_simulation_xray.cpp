#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <random>
#include <st_handeye/st_handeye.hpp>


class HandeyeSimulationXray {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    HandeyeSimulationXray(const Eigen::MatrixXd& camera_matrix, const Eigen::MatrixXd& pattern_3d, const Eigen::Isometry3d& hand2detector, const Eigen::Vector3d& hand2source, const Eigen::Isometry3d& object2world)
        : camera_matrix(camera_matrix),
          pattern_3d(pattern_3d),
          hand2detector(hand2detector),
          hand2source(hand2source),
          object2world(object2world)
    {}

    Eigen::MatrixXd project(const Eigen::Isometry3d& world2hand_detector, const Eigen::Isometry3d& world2hand_source) const {
        Eigen::Vector3d source = hand2detector * world2hand_detector * world2hand_source.inverse() * -hand2source;
        Eigen::Isometry3d object2detector = hand2detector * world2hand_detector * object2world;

        Eigen::MatrixXd pattern = object2detector.matrix() * pattern_3d;

        Eigen::ArrayXd t = source.z() / (source.z() - pattern.row(2).array());
        Eigen::MatrixXd rays = (pattern.colwise() - source);
        Eigen::MatrixXd offsets = rays.array().rowwise() * t.transpose();
        Eigen::MatrixXd projected = offsets.matrix().colwise() + source;

        std::cout << pattern.transpose() << std::endl;
        std::cout << "source " << source.transpose() << std::endl;

        projected.row(2).setOnes();

        Eigen::MatrixXd uvs = camera_matrix * projected;

        return uvs.topRows(2);
    }

    cv::Mat visualize(const Eigen::MatrixXd& pattern_2d, int sleep=100) const {
        cv::Mat canvas(480, 640, CV_8UC3, cv::Scalar::all(0));
        for(int i=0; i<pattern_2d.cols(); i++) {
            cv::Point pt(pattern_2d(0, i), pattern_2d(1, i));
            cv::circle(canvas, pt, 5, cv::Scalar(255, 0, 0), -1);
        }

        cv::imshow("canvas", canvas);
        cv::waitKey(sleep);
        return canvas;
    }

private:
    Eigen::MatrixXd camera_matrix;
    Eigen::MatrixXd pattern_3d;

    Eigen::Vector3d hand2source;
    Eigen::Isometry3d hand2detector;
    Eigen::Isometry3d object2world;
};


int main(int argc, char** argv) {
    using namespace boost::program_options;

    options_description description("handeye_simulation_xray");
    description.add_options()
            ("seed,s", value<long>()->default_value(0), "seed of random")
            ("visualize,v", value<bool>()->default_value(false), "if visualize")
            ("fx", value<double>()->default_value(200), "focal length")
            ("x_steps", value<int>()->default_value(1), "x steps")
            ("y_steps", value<int>()->default_value(1), "y steps")
            ("z_steps", value<int>()->default_value(1), "z steps")
            ("x_step", value<double>()->default_value(0.5), "x step")
            ("y_step", value<double>()->default_value(0.5), "y step")
            ("z_step", value<double>()->default_value(2.0), "z step")
            ("z_offset", value<double>()->default_value(2.0), "z offset")
            ("hand2detector_trans", value<double>()->default_value(0.1), "stddev of the translation of the handeye transformation")
            ("hand2detector_rot", value<double>()->default_value(10.0), "stddev of the rotation angle [deg] of the handeye transformation")
            ("hand2source_trans", value<double>()->default_value(0.3), "stddev of the translation of the hand2source transformation")
            ("tnoise", value<double>()->default_value(0.0), "stddev of the translation noise of the world2hand transformation")
            ("rnoise", value<double>()->default_value(0.0), "stddev of the rotation noise [deg] of the world2hand transformation")
            ("vnoise", value<double>()->default_value(0.0), "stddev of the visual noise on the marker detection")
            ("visual_inf_scale", value<double>()->default_value(1.0), "scale of the informatoin matrix for visual detections")
            ("source_inf_scale", value<double>()->default_value(1.0), "scale of the informatoin matrix for visual detections")
            ("handpose_inf_scale_trans", value<double>()->default_value(1.0), "scale of the informatoin matrix for hand poses")
            ("handpose_inf_scale_rot", value<double>()->default_value(1.0), "scale of the informatoin matrix for hand poses")
            ("num_iterations", value<int>()->default_value(8192), "max number of iterations")
            ("solver_name", value<std::string>()->default_value("lm_var_cholmod"), "g2o solver name")
            ("robust_kernel_handpose", value<std::string>()->default_value("NONE"), "robust kernel for handpose edges")
            ("robust_kernel_projection", value<std::string>()->default_value("NONE"), "robust kernel for projection edges")
            ("robust_kernel_source", value<std::string>()->default_value("NONE"), "robust kernel for projection edges")
            ("robust_kernel_handpose_delta", value<double>()->default_value(1.0), "robust kernel delta for handpose edges")
            ("robust_kernel_projection_delta", value<double>()->default_value(1.0), "robust kernel delta for projection edges")
            ("robust_kernel_source_delta", value<double>()->default_value(1.0), "robust kernel delta for projection edges")
    ;

    variables_map vm;
    store(parse_command_line(argc, argv, description), vm);
    notify(vm);

    std::mt19937 mt(vm["seed"].as<long>());
    srand(mt());

    auto transnoise = [&]() {
        double tnoise = std::normal_distribution<>(0.0, vm["tnoise"].as<double>())(mt);
        double rnoise = std::normal_distribution<>(0.0, vm["rnoise"].as<double>())(mt) * M_PI / 180.0;

        Eigen::Isometry3d noise = Eigen::Isometry3d::Identity();
        if(tnoise > 1e-9) {
            noise.translation() = (Eigen::Vector3d::Random() - Eigen::Vector3d::Ones() * 0.5).normalized() * tnoise;
        }
        if(rnoise > 1e-9) {
            Eigen::Vector3d axis = (Eigen::Vector3d::Random() - Eigen::Vector3d::Ones() * 0.5).normalized();
            noise.linear() = Eigen::AngleAxisd(rnoise, axis).toRotationMatrix();
        }

        return noise;
    };


    Eigen::Matrix3d camera_matrix = Eigen::Matrix3d::Identity();
    camera_matrix(0, 0) = vm["fx"].as<double>();
    camera_matrix(1, 1) = vm["fx"].as<double>();
    camera_matrix(0, 2) = 320.0;
    camera_matrix(1, 2) = 240.0;

    int rows = 7;
    int cols = 5;
    double dimension = 0.1;
    Eigen::MatrixXd pattern_3d(4, rows * cols);
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            pattern_3d.col(i * cols + j) = Eigen::Vector4d(i - (rows - 1) / 2.0, j - (cols - 1)/2.0, 0.0, 1);
            pattern_3d.col(i * cols + j).topRows(3) *= dimension;
        }
    }

    // hand2detector
    double hand2detector_trans = std::normal_distribution<>(0.0, vm["hand2detector_trans"].as<double>())(mt);
    double hand2detector_rotangle = std::normal_distribution<>(0.0, vm["hand2detector_rot"].as<double>())(mt) * M_PI / 180.0;
    Eigen::Vector3d hand2detector_rotaxis = (Eigen::Vector3d::Random() - Eigen::Vector3d::Ones() * 0.5).normalized();

    Eigen::Isometry3d hand2detector = Eigen::Isometry3d::Identity();
    hand2detector.translation() = (Eigen::Vector3d::Random() - Eigen::Vector3d::Ones() * 0.5).normalized() * hand2detector_trans;
    hand2detector.linear() = Eigen::AngleAxisd(hand2detector_rotangle, hand2detector_rotaxis).toRotationMatrix();

    // hand2source
    double hand2source_trans = std::normal_distribution<>(0.0, vm["hand2source_trans"].as<double>())(mt);
    Eigen::Vector3d hand2source = (Eigen::Vector3d::Random() - Eigen::Vector3d::Ones() * 0.5).normalized() * hand2source_trans;

    // object2world
    Eigen::Isometry3d object2world = Eigen::Isometry3d::Identity();
    object2world.translation() = Eigen::Vector3d(0.0, 0.0, 0.0);

    //
    std::unique_ptr<HandeyeSimulationXray> sim(new HandeyeSimulationXray(camera_matrix, pattern_3d, hand2detector, hand2source, object2world));

    std::vector<Eigen::Isometry3d> world2hands_detector;
    std::vector<Eigen::Isometry3d> world2hands_source;
    std::vector<Eigen::MatrixXd> pattern_2ds;

    int x_steps = vm["x_steps"].as<int>();
    int y_steps = vm["y_steps"].as<int>();
    int z_steps = vm["z_steps"].as<int>();

    // fix source and move detector
    for(int z_step=0; z_step<=z_steps; z_step++) {
        for(int y_step=-y_steps; y_step<=y_steps; y_step++) {
            for(int x_step=-x_steps; x_step<=x_steps; x_step++) {
                // world2hand_source
                double source_hand_rotangle = std::uniform_real_distribution<>(0.0, 90.0)(mt);
                Eigen::Vector3d source_hand_rotaxis = (Eigen::Vector3d::Random() - Eigen::Vector3d::Ones() * 0.5).normalized();

                Eigen::Isometry3d source_handpose = Eigen::Isometry3d::Identity();
                source_handpose.translation() = Eigen::Vector3d(0.0, 0.0, vm["z_offset"].as<double>());
                source_handpose.linear() = Eigen::AngleAxisd(source_hand_rotangle, source_hand_rotaxis).toRotationMatrix();
                source_handpose.translation() += source_handpose.linear() * hand2source;

                Eigen::Isometry3d world2hand_source = source_handpose.inverse();


                // word2hand_detector
                Eigen::Isometry3d detector_handpose = Eigen::Isometry3d::Identity();
                detector_handpose.translation().x() = x_step * vm["x_step"].as<double>();
                detector_handpose.translation().y() = y_step * vm["y_step"].as<double>();
                detector_handpose.translation().z() = -vm["z_offset"].as<double>() - z_step * vm["z_step"].as<double>();

                Eigen::Vector3d dir_from = detector_handpose.linear().col(2);
                Eigen::Vector3d dir_to = (object2world.translation() - detector_handpose.translation()).normalized();

                double angle = std::acos(dir_from.dot(dir_to));
                if(angle > 1e-6) {
                    detector_handpose.linear() = Eigen::AngleAxisd(angle, dir_from.cross(dir_to).normalized()) * detector_handpose.linear();
                }

                Eigen::Isometry3d world2hand_detector = hand2detector.inverse() * detector_handpose.inverse();

                // rendering
                Eigen::MatrixXd pattern_2d = sim->project(world2hand_detector, world2hand_source);
                if(vm["visualize"].as<bool>()) {
                    cv::Mat image = sim->visualize(pattern_2d, 100);
                }

                double vnoise = vm["vnoise"].as<double>();
                if(vnoise > 1e-9) {
                    for(int k=0; k<pattern_2d.cols(); k++) {
                        double noisemag = std::normal_distribution<>(0.0, vnoise)(mt);
                        Eigen::Vector2d noisevec = (Eigen::Vector2d::Random() - Eigen::Vector2d::Ones() * 0.5).normalized();
                        pattern_2d.col(k) += noisevec * noisemag;
                    }
                }

                world2hands_detector.push_back(transnoise() * world2hand_detector);
                world2hands_source.push_back(transnoise() * world2hand_source);
                pattern_2ds.push_back(pattern_2d);
            }
        }
    }

    // fix detector and move source
    for(int z_step=0; z_step<=z_steps; z_step++) {
        for(int y_step=-y_steps; y_step<=y_steps; y_step++) {
            for(int x_step=-x_steps; x_step<=x_steps; x_step++) {
                // word2hand_detector
                Eigen::Isometry3d detector_handpose = Eigen::Isometry3d::Identity();
                detector_handpose.translation().z() = -vm["z_offset"].as<double>();
                Eigen::Isometry3d world2hand_detector = hand2detector.inverse() * detector_handpose.inverse();

                // world2hand_source
                double source_hand_rotangle = std::uniform_real_distribution<>(0.0, 90.0)(mt);
                Eigen::Vector3d source_hand_rotaxis = (Eigen::Vector3d::Random() - Eigen::Vector3d::Ones() * 0.5).normalized();

                Eigen::Isometry3d source_handpose = Eigen::Isometry3d::Identity();
                source_handpose.translation().x() = x_step * vm["x_step"].as<double>();
                source_handpose.translation().y() = y_step * vm["y_step"].as<double>();
                source_handpose.translation().z() = vm["z_offset"].as<double>() + z_step * vm["z_step"].as<double>();

                source_handpose.linear() = Eigen::AngleAxisd(source_hand_rotangle, source_hand_rotaxis).toRotationMatrix();
                source_handpose.translation() += source_handpose.linear() * hand2source;

                Eigen::Isometry3d world2hand_source = source_handpose.inverse();

                // rendering
                Eigen::MatrixXd pattern_2d = sim->project(world2hand_detector, world2hand_source);
                if(vm["visualize"].as<bool>()) {
                    cv::Mat image = sim->visualize(pattern_2d, 100);
                }

                double vnoise = vm["vnoise"].as<double>();
                if(vnoise > 1e-9) {
                    for(int k=0; k<pattern_2d.cols(); k++) {
                        double noisemag = std::normal_distribution<>(0.0, vnoise)(mt);
                        Eigen::Vector2d noisevec = (Eigen::Vector2d::Random() - Eigen::Vector2d::Ones() * 0.5).normalized();
                        pattern_2d.col(k) += noisevec * noisemag;
                    }
                }

                world2hands_detector.push_back(transnoise() * world2hand_detector);
                world2hands_source.push_back(transnoise() * world2hand_source);
                pattern_2ds.push_back(pattern_2d);
            }
        }
    }

    // fix detector and move source
    for(int z_step=0; z_step<=z_steps; z_step++) {
        for(int y_step=-y_steps; y_step<=y_steps; y_step++) {
            for(int x_step=-x_steps; x_step<=x_steps; x_step++) {
                // world2hand_source
                double source_hand_rotangle = std::uniform_real_distribution<>(0.0, 90.0)(mt);
                Eigen::Vector3d source_hand_rotaxis = (Eigen::Vector3d::Random() - Eigen::Vector3d::Ones() * 0.5).normalized();

                Eigen::Vector3d source_world_pos;
                source_world_pos.x() = x_step * vm["x_step"].as<double>();
                source_world_pos.y() = y_step * vm["y_step"].as<double>();
                source_world_pos.z() = vm["z_offset"].as<double>() + z_step * vm["z_step"].as<double>();

                Eigen::Isometry3d source_handpose = Eigen::Isometry3d::Identity();
                source_handpose.translation() = source_world_pos;
                source_handpose.linear() = Eigen::AngleAxisd(source_hand_rotangle, source_hand_rotaxis).toRotationMatrix();
                source_handpose.translation() += source_handpose.linear() * hand2source;

                Eigen::Isometry3d world2hand_source = source_handpose.inverse();


                // word2hand_detector
                Eigen::Isometry3d detector_handpose = Eigen::Isometry3d::Identity();

                Eigen::Vector3d source2object = (object2world.translation() - source_world_pos).normalized();
                double unit_t = 1.0 / source2object.z();
                double t = (source_world_pos.z() * 2) * unit_t;
                detector_handpose.translation() = source_world_pos - source2object * t;

                Eigen::Vector3d dir_from = detector_handpose.linear().col(2);
                Eigen::Vector3d dir_to = -source2object;

                double angle = std::acos(dir_from.dot(dir_to));
                if(angle > 1e-9) {
                    Eigen::Vector3d axis = dir_from.cross(dir_to).normalized();
                    detector_handpose.linear() = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
                }

                Eigen::Isometry3d world2hand_detector = hand2detector.inverse() * detector_handpose.inverse();

                // rendering
                Eigen::MatrixXd pattern_2d = sim->project(world2hand_detector, world2hand_source);
                if(vm["visualize"].as<bool>()) {
                    cv::Mat image = sim->visualize(pattern_2d, 100);
                }

                double vnoise = vm["vnoise"].as<double>();
                if(vnoise > 1e-9) {
                    for(int k=0; k<pattern_2d.cols(); k++) {
                        double noisemag = std::normal_distribution<>(0.0, vnoise)(mt);
                        Eigen::Vector2d noisevec = (Eigen::Vector2d::Random() - Eigen::Vector2d::Ones() * 0.5).normalized();
                        pattern_2d.col(k) += noisevec * noisemag;
                    }
                }

                world2hands_detector.push_back(transnoise() * world2hand_detector);
                world2hands_source.push_back(transnoise() * world2hand_source);
                pattern_2ds.push_back(pattern_2d);
            }
        }
    }

    std::cout << "num_images: " << world2hands_detector.size() << std::endl;
    std::cout << "calibrating..." << std::endl;

    Eigen::Isometry3d hand2detector_ = hand2detector;
    Eigen::Vector3d hand2source_ = hand2source;
    Eigen::Isometry3d object2world_ = object2world;

    hand2detector_.setIdentity();
    hand2source_.setZero();
    object2world_.setIdentity();

    st_handeye::OptimizationParams params;
    params.source_inf_scale = vm["source_inf_scale"].as<double>();
    params.pattern2d_inf_scale = vm["visual_inf_scale"].as<double>();
    params.world2hand_inf_scale_trans = vm["handpose_inf_scale_trans"].as<double>();
    params.world2hand_inf_scale_rot = vm["handpose_inf_scale_rot"].as<double>();
    params.num_iterations = vm["num_iterations"].as<int>();
    params.solver_name = vm["solver_name"].as<std::string>();
    params.robust_kernel_handpose = vm["robust_kernel_handpose"].as<std::string>();
    params.robust_kernel_projection = vm["robust_kernel_projection"].as<std::string>();
    params.robust_kernel_source = vm["robust_kernel_source"].as<std::string>();
    params.robust_kernel_handpose_delta = vm["robust_kernel_handpose_delta"].as<double>();
    params.robust_kernel_projection_delta = vm["robust_kernel_projection_delta"].as<double>();
    params.robust_kernel_source_delta = vm["robust_kernel_source_delta"].as<double>();

    if(!st_handeye::spatial_calibration_graph_xray(camera_matrix, pattern_3d, world2hands_detector, world2hands_source, pattern_2ds, hand2detector_, hand2source_, object2world_, params)){
        std::cerr << "failed to calibrate..." << std::endl;
        return 1;
    }

    std::cout << "--- hand2detector ---\n" << hand2detector.matrix() << std::endl;
    std::cout << "--- hand2detector_ ---\n" << hand2detector_.matrix() << std::endl;

    std::cout << "--- hand2source ---\n" << hand2source.transpose() << std::endl;
    std::cout << "--- hand2source_ ---\n" << hand2source_.transpose() << std::endl;

    std::cout << "--- object2world ---\n" << object2world.matrix() << std::endl;
    std::cout << "--- object2world_ ---\n" << object2world_.matrix() << std::endl;

    Eigen::Isometry3d delta_detector = hand2detector.inverse() * hand2detector_;
    double terror_detector = delta_detector.translation().norm();
    double rerror_detector = Eigen::AngleAxisd().fromRotationMatrix(delta_detector.linear()).angle() * 180.0 / M_PI;

    Eigen::Vector3d delta_source = hand2source - hand2source_;

    std::cout << "terror_detector " << terror_detector << " rerror_detector " << rerror_detector << " terror_source " << delta_source.norm() << std::endl;

    return 0;
}
