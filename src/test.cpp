#include <iostream>
#include <Eigen/Dense>
#include <st_handeye/st_handeye.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/aruco/charuco.hpp>

int main(int argc, char const *argv[])
{
	int PATTERN_ROWS = 9;
	int PATTERN_COLS = 13;

	cv::Mat image = cv::imread("/home/jingyixiang/test/calib_images/current/003_image.jpg");
	std::cout << image.cols << ", " << image.rows << std::endl;
	// cv::resize(image, image, cv::Size(image.cols * 0.8,image.rows * 0.8), 0, 0, CV_INTER_LINEAR);
	// gray = cv2.cvtColor(chess_board_image, cv2.COLOR_RGB2GRAY)
	cv::Mat gray = image;
	// cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);

	cv::Mat cv_grid_2d;
	bool ret = cv::findChessboardCorners(gray, cv::Size(PATTERN_ROWS, PATTERN_COLS), cv_grid_2d, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
	// bool ret = cv::findCirclesGrid(gray, cv::Size(PATTERN_ROWS, PATTERN_COLS), cv_grid_2d, cv::CALIB_CB_ASYMMETRIC_GRID);
	cv::drawChessboardCorners(gray, cv::Size(PATTERN_ROWS, PATTERN_COLS), cv_grid_2d, ret);
	std::cout << ret << std::endl;
	cv::imwrite("/home/jingyixiang/test.jpg", gray);

	return 0;


	// // charuco test
	// std::cout << "OpenCV version : " << CV_VERSION << std::endl;

	// cv::Mat image = cv::imread("/home/jingyixiang/test/charuco_image.jpg");

 //    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
 //    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 7, 0.04f, 0.03f, dictionary);
 //    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();

 //    std::vector<int> markerIds;
 //    std::vector<std::vector<cv::Point2f> > markerCorners;
 //    cv::aruco::detectMarkers(image, board->dictionary, markerCorners, markerIds, params);

 //    std::vector<cv::Point2f> charucoCorners;
 //    std::vector<int> charucoIds;
 //    cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, image, board, charucoCorners, charucoIds);

 //    cv::Mat imageCopy = image;
 //    cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds);
 //    cv::imwrite("/home/jingyixiang/test/test.jpg", imageCopy);
}


// // world2hands[i] = Eigen::Isometry3d(dataset->handposes[i]).inverse();
// // camera_param->setKcam(camera_matrix(0, 0), camera_matrix(1, 1), camera_matrix(0, 2), camera_matrix(1, 2));
// // void g2o::ParameterCamera::setKcam(double fx, double fy, double cx, double cy)

// Eigen::Isometry3d iso3d_from_arr (double arr[12]){
// 	Eigen::Isometry3d ret = Eigen::Isometry3d::Identity();

// 	for(int i = 0; i < 3; i++){
// 		for(int j = 0; j < 4; j++){
// 			ret(i, j) = arr[i*4 + j];
// 		}
// 	}

// 	return ret;
// }

// Eigen::MatrixXd mat_xd_from_arr (double arr[40]){
// 	Eigen::MatrixXd ret(2, 20);
// 	for(int i = 0; i < 2; i++){
// 		for(int j = 0; j < 20; j++){
// 			ret(i, j) = arr[i*20 + j];
// 		}
// 	}

// 	return ret;
// }

// int main(int argc, char const *argv[])
// {
// 	// camera matrix
// 	Eigen::Matrix3d camera_matrix;
// 	camera_matrix << 612.2394409179688, 			  0.0, 323.92718505859375,
// 					 			   0.0, 610.8439331054688, 236.01596069335938,
// 					 			   0.0, 			  0.0, 				  1.0;
// 	// std::cout << camera_matrix << std::endl;

// 	// we have 20 points - so 3*20
// 	Eigen::MatrixXd pattern_3d(3, 20);
// 	// start from top left
// 	// pattern_3d << 0.0,  0.04,  0.08,  0.12,  0.16,   0.0,  0.04,  0.08,  0.12,  0.16,   0.0,  0.04,  0.08,  0.12,  0.16,   0.0,  0.04,  0.08,  0.12,  0.16,
// 	// 			  0.0,   0.0,   0.0,   0.0,   0.0, -0.04, -0.04, -0.04, -0.04, -0.04, -0.08, -0.08, -0.08, -0.08, -0.08, -0.12, -0.12, -0.12, -0.12, -0.12,
// 	// 			  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0;
// 	// start from bottom right
// 	pattern_3d << 0.16,  0.12,  0.08,  0.04,   0.0,  0.16,  0.12,  0.08,  0.04,   0.0,  0.16,  0.12,  0.08,  0.04,   0.0,  0.16,  0.12,  0.08,  0.04,   0.0,
// 				   0.0,   0.0,   0.0,   0.0,   0.0,  0.04,  0.04,  0.04,  0.04,  0.04,  0.08,  0.08,  0.08,  0.08,  0.08,  0.12,  0.12,  0.12,  0.12,  0.12,
// 				   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0;
// 	// std::cout << pattern_3d << std::endl;
// 	// std::cout << pattern_3d.rows() << ", " << pattern_3d.cols() << std::endl;

// 	// world2hands
// 	// tool0 base_link
// 	// double img8_hand_arr[12] = { 0.000500401321860116,    -0.894169123637157,  -0.4477290787211344,   0.042, 
// 	// 						      -0.9978522775265763, -0.029773878650677838, 0.058346794128891405,   0.272, 
// 	// 						     -0.06550253303149128,    0.4467382841038513,  -0.8922635954035136,   0.359};
// 	// double img9_hand_arr[12] = { 0.017753456786046916,  -0.8903584211100339,  0.45491394431320054, -0.001, 
// 	// 						 -0.9986141837542313, 0.006754978682444034, 0.052192741524758085,  0.279, 
// 	// 					   -0.049543180933569936, -0.45521011875994966,  -0.8890046237233732,  0.376};
// 	// double img10_hand_arr[12] = {-0.0005129246000837862,    -0.9999990001469787, -0.0013178062824764636,   0.0, 
// 	// 							    -0.7368696801570171, -0.0005129246000837862,     0.6760346229104324, 0.193, 
// 	// 							    -0.6760346229104324,  0.0013178062824764636,    -0.7368686803039957, 0.507};
// 	// base_link tool0
// 	double img8_hand_arr[12] = {0.000500401321860116, -0.9978522775265763, -0.06550253303149128, 0.295, -0.894169123637157, -0.029773878650677838, 0.4467382841038513, -0.114, -0.4477290787211344, 0.058346794128891405, -0.8922635954035136, 0.323};
// 	double img9_hand_arr[12] = {0.017753456786046916, -0.9986141837542313, -0.049543180933569936, 0.297, -0.8903584211100339, 0.006754978682444034, -0.45521011875994966, 0.168, 0.45491394431320054, 0.052192741524758085, -0.8890046237233732, 0.32};
// 	double img10_hand_arr[12] = {0.0, -0.7359782543521277, -0.6770051765834549, 0.485, -1.0, 0.0, 0.0, 0.0, 0.0, 0.6770051765834549, -0.7359782543521277, 0.243};
// 	std::vector<Eigen::Isometry3d> world2hands;
// 	world2hands.push_back(iso3d_from_arr(img8_hand_arr));
// 	world2hands.push_back(iso3d_from_arr(img9_hand_arr));
// 	world2hands.push_back(iso3d_from_arr(img10_hand_arr));

// 	// corners
// 	double img8_corners_arr[40] = {1601.9769, 1394.5, 1208.6324, 1045.6104, 898.44617, 1611.125, 1400.2865, 1213.4122, 1049.4188, 901.3322, 1621.1697, 1406.794, 1218.2574, 1051.6824, 902.4677, 1629.7745, 1415.0621, 1223.2388, 1053.8313, 901.72284,
// 								   836.9112, 810.0, 784.3897, 762.75586, 744.59973, 638.7994, 622.863, 609.31396, 597.32556, 587.49695, 437.57944, 434.29822, 432.2985, 430.42697, 428.06635, 230.75467, 240.7808, 250.40813, 259.0571, 265.78195};
// 	double img9_corners_arr[40] = {1321.6039, 1195.2509, 1055.673, 899.5172, 721.44165, 1324.2319, 1197.3911, 1057.2938, 901.4005, 723.4714, 1330.4496, 1203.1107, 1062.8037, 904.72955, 726.3157, 1339.405, 1210.3044, 1067.5846, 909.2779, 729.7075,
// 								   779.2254, 792.1627, 807.2081, 825.37915, 847.70654, 624.6378, 630.00214, 636.06995, 644.1102, 653.154, 470.3254, 467.37497, 464.0718, 461.48615, 457.75198, 312.27008, 301.6092, 290.51862, 276.34995, 257.83893};
// 	double img10_corners_arr[40] = {1372.5039, 1196.7274, 1021.66724, 846.90936, 670.35486, 1412.7727, 1219.8759, 1029.0946, 838.41626, 645.5807, 1463.7933, 1249.1338, 1038.3704, 827.76, 613.6315, 1526.838, 1287.6974, 1050.7799, 813.8904, 570.5,
// 									709.1229, 706.9998, 706.07196, 704.7639, 706.54803, 583.29675, 581.36115, 579.2206, 579.16174, 579.13153, 434.63007, 433.48486, 431.38812, 430.22842, 429.22528, 252.60008, 250.1904, 249.23454, 248.09703, 244.0};
// 	std::vector<Eigen::MatrixXd> pattern_2ds;
// 	pattern_2ds.push_back(mat_xd_from_arr(img8_corners_arr));
// 	pattern_2ds.push_back(mat_xd_from_arr(img9_corners_arr));
// 	pattern_2ds.push_back(mat_xd_from_arr(img10_corners_arr));

// 	// hand2eye
// 	double hand_eye_arr[12] = {   0.9999855708190154, -0.0032077924602275723, -0.004309085900801766, -0.028505236989580417, 
// 						   	   0.0032323686208022926,      0.999978482008643,  0.005708529827344272, -0.013078112032390567, 
// 						        0.004290681398989456,  -0.005722376011984328,    0.9999744219058357,  0.032377136598431805};
// 	Eigen::Isometry3d hand2eye = iso3d_from_arr(hand_eye_arr);

// 	// object2world
// 	// this could be wrong
// 	// this is using base_link charuco. 
// 	double object_world_arr[12] = {0.9986312455665344, 0.006773835809413434, 0.05186280484758869, 0.345, -0.005215254118751935, 0.9995324254928015, -0.030128582989479572, 0.103, -0.05204264119651117, 0.029816866651347276, 0.9981996383291204, 0.002};
// 	// this is using charuco base_link
// 	// double object_world_arr[12] = {0.9987330882771702, -0.005245494120550448, -0.050047009618737516, -0.345, 0.006744206726422005, 0.9995324016669683, 0.029824380856843975, -0.104, 0.04986716410603293, -0.030124123378018286, 0.9983014590466792, -0.013};
// 	Eigen::Isometry3d object2world = iso3d_from_arr(object_world_arr);

// 	std::cout << "error free!" << std::endl;

// 	st_handeye::OptimizationParams params = st_handeye::OptimizationParams();
// 	bool c = st_handeye::spatial_calibration_graph (camera_matrix, pattern_3d, world2hands, pattern_2ds, hand2eye, object2world, params);
// 	std::cout << c << std::endl;

// 	std::cout << "\n" << "hand to eye: \n";
// 	for(int i = 0; i < 3; i++){
// 		for(int j = 0; j < 4; j++){
// 			std::cout << hand2eye(i, j) << ", ";
// 		}
// 		std::cout << "\n";
// 	}

// 	std::cout << "\n" << "object to world: \n";
// 	for(int i = 0; i < 3; i++){
// 		for(int j = 0; j < 4; j++){
// 			std::cout << object2world(i, j) << ", ";
// 		}
// 		std::cout << "\n";
// 	}

// 	return 0;
// }
