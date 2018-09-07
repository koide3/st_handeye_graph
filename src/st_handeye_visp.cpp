#include <regex>
#include <random>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

#include <visp/vpDebug.h>
#include <visp/vpPoint.h>
#include <visp/vpCalibration.h>
#include <visp/vpExponentialMap.h>

#include <st_handeye/st_handeye.hpp>


namespace st_handeye {

cv::Mat eigen2cvmat(const Eigen::MatrixXd& matrix) {
    cv::Mat cv_mat(matrix.rows(), matrix.cols(), CV_64FC1);
    for(int i=0; i<matrix.rows(); i++) {
        for(int j=0; j<matrix.cols(); j++) {
            cv_mat.at<double>(i, j) = matrix(i, j);
        }
    }
    return cv_mat;
}

Eigen::MatrixXd cvmat2eigen(const cv::Mat& matrix) {
    Eigen::MatrixXd eigen_mat(matrix.rows, matrix.cols);
    for(int i=0; i<matrix.rows; i++) {
        for(int j=0; j<matrix.cols; j++) {
            eigen_mat(i, j) = matrix.at<double>(i, j);
        }
    }
    return eigen_mat;
}

vpHomogeneousMatrix eigen2vpmat(const Eigen::MatrixXd& matrix) {
    vpHomogeneousMatrix vpmat;
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            vpmat[i][j] = matrix(i, j);
        }
    }
    return vpmat;
}

Eigen::MatrixXd vpmat2eigen(const vpHomogeneousMatrix& matrix) {
    Eigen::MatrixXd eigen_mat(4, 4);
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            eigen_mat(i, j) = matrix[i][j];
        }
    }
    return eigen_mat;
}

std::vector<Eigen::Isometry3d> calc_object2eyes(
        const Eigen::Matrix3d& camera_matrix,
        const Eigen::MatrixXd& pattern_3d,
        const std::vector<Eigen::MatrixXd>& pattern_2ds
) {
    cv::Mat cv_camera_matrix = eigen2cvmat(camera_matrix);
    cv::Mat cv_pattern_3d = eigen2cvmat(pattern_3d);

    std::vector<Eigen::Isometry3d> object2eyes(pattern_2ds.size());
    for(int i=0; i<pattern_2ds.size(); i++) {
        cv::Mat cv_pattern_2d = eigen2cvmat(pattern_2ds[i]);

        cv::Mat rvec, tvec;
        cv::solvePnP(cv_pattern_3d.t(), cv_pattern_2d.t(), cv_camera_matrix, cv::Mat(), rvec, tvec);

        cv::Mat rotation;
        cv::Rodrigues(rvec, rotation);

        Eigen::Isometry3d object2eye = Eigen::Isometry3d::Identity();
        object2eye.translation() = cvmat2eigen(tvec);
        object2eye.linear() = cvmat2eigen(rotation);

        object2eyes[i] = object2eye;
    }

    return object2eyes;
}

bool spatial_calibration_visp (
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::MatrixXd& pattern_3d,
    const std::vector<Eigen::Isometry3d>& world2hands,
    const std::vector<Eigen::MatrixXd>& pattern_2ds,
    Eigen::Isometry3d& hand2eye,
    Eigen::Isometry3d& object2world,
    OptimizationParams params
) {
    std::vector<Eigen::Isometry3d> object2eyes = calc_object2eyes(camera_matrix, pattern_3d, pattern_2ds);

    std::vector<vpHomogeneousMatrix> vp_eye2objects;
    std::vector<vpHomogeneousMatrix> vp_world2hands;

    for(int i=0; i<world2hands.size(); i++) {
        vp_eye2objects.push_back(eigen2vpmat(object2eyes[i].matrix()));
        vp_world2hands.push_back(eigen2vpmat(world2hands[i].inverse().matrix()));
    }

    vpHomogeneousMatrix vp_hand2eye = eigen2vpmat(hand2eye.matrix());
    vpCalibration::calibrationTsai(vp_eye2objects, vp_world2hands, vp_hand2eye);

    Eigen::Matrix4d hand2eye_ = vpmat2eigen(vp_hand2eye).inverse();
    hand2eye.translation() = hand2eye_.block<3, 1>(0, 3);
    hand2eye.linear() = hand2eye_.block<3, 3>(0, 0);

    object2world = world2hands[0].inverse() * hand2eye.inverse() * object2eyes[0];

    return true;
}

bool spatial_calibration_dualquaternion (
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::MatrixXd& pattern_3d,
    const std::vector<Eigen::Isometry3d>& world2hands,
    const std::vector<Eigen::MatrixXd>& pattern_2ds,
    Eigen::Isometry3d& hand2eye,
    Eigen::Isometry3d& object2world,
    OptimizationParams params
) {
    std::vector<Eigen::Isometry3d> object2eyes = calc_object2eyes(camera_matrix, pattern_3d, pattern_2ds);
    long pid = getpid();
    long rnd = random();
    std::string input_filename = (boost::format("/tmp/poses_%d_%d.yml") % pid % rnd).str();
    std::string output_filename = (boost::format("/tmp/handeye_%d_%d.yml") % pid % rnd).str();
    std::string nodename = (boost::format("handeye_calib_camodocal_%d_%d") % pid % rnd).str();

    cv::FileStorage fs(input_filename, cv::FileStorage::WRITE);
    fs << "frameCount" << static_cast<int>(world2hands.size());

    for(int i=0; i<world2hands.size(); i++) {
        cv::Mat cv_world2hand = eigen2cvmat(world2hands[i].inverse().matrix());
        cv::Mat cv_object2eye = eigen2cvmat(object2eyes[i].inverse().matrix());

        fs << (boost::format("T1_%d") % i).str() << cv_world2hand;
        fs << (boost::format("T2_%d") % i).str() << cv_object2eye;
    }

    fs.release();

    std::stringstream sst;
    sst << "rosrun handeye_calib_camodocal handeye_calib_camodocal "
        << "__name:=" << nodename << " "
        << "_load_transforms_from_file:=true "
        << "_transform_pairs_load_filename:=" << input_filename << " "
        << "_output_calibrated_transform_filename:=" << output_filename;

    std::string command = sst.str();
    std::cout << "command: " << command << std::endl;

    if(system(command.c_str())) {
        return false;
    }

    cv::FileStorage ifs(output_filename, cv::FileStorage::READ);
    cv::Mat cv_hand2eye;
    ifs["ArmTipToMarkerTagTransform"] >> cv_hand2eye;

    Eigen::Matrix4d h2e = cvmat2eigen(cv_hand2eye);
    hand2eye = Eigen::Isometry3d(h2e).inverse();

    object2world = world2hands[0].inverse() * hand2eye.inverse() * object2eyes[0];

    return true;
}

}
