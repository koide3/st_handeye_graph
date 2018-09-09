#!/usr/bin/python
import os
import re
import cv2
import numpy
import argparse
import scipy.optimize


def read_params(args):
	image_files = sorted([x for x in os.listdir(args.directory) if '_image.jpg' in x])
	data_ids = [x[:3] for x in image_files]

	if args.ros_camera_params is not None:
		with open(args.ros_camera_params, 'r') as f:
			while 'camera_matrix' not in f.readline():
				pass

			rows = int(re.search(r'rows:\s*([0-9]+)', f.readline()).group(1))
			cols = int(re.search(r'cols:\s*([0-9]+)', f.readline()).group(1))
			values = re.search(r'data:\s*\[(.+)\]', f.readline()).group(1)
			values = [x.strip() for x in values.split(',')]

			assert(rows == 3 and cols == 3 and len(values) == 9)
			camera_matrix = numpy.float32(values).reshape(rows, cols)

			while 'distortion_coefficients' not in f.readline():
				pass

			rows = int(re.search(r'rows:\s*([0-9]+)', f.readline()).group(1))
			cols = int(re.search(r'cols:\s*([0-9]+)', f.readline()).group(1))
			values = re.search(r'data:\s*\[(.+)\]', f.readline()).group(1)
			values = [x.strip() for x in values.split(',')]

			assert(rows == 1 and cols == 5 and len(values) == 5)
			distortion = numpy.float32(values)
	else:
		camera_matrix = numpy.loadtxt('%s/000_camera_matrix.csv' % args.directory)
		distortion = numpy.loadtxt('%s/000_distortion.csv' % args.directory)

	return {'data_ids': data_ids, 'camera_matrix': camera_matrix, 'distortion': distortion}


def undistort(params, args):
	camera_matrix = params['camera_matrix']
	distortion = params['distortion']
	data_ids = params['data_ids']
	images = [cv2.imread('%s/%s_image.jpg' % (args.directory, x)) for x in data_ids]

	for data_id, image in zip(data_ids, images):
		undistorted = cv2.undistort(image, camera_matrix, distortion)
		cv2.imwrite('%s/%s_undistorted.jpg' % (args.directory, data_id), undistorted)
		cv2.imshow('undistorted', cv2.resize(undistorted, (undistorted.shape[1] / 2, undistorted.shape[0] / 2)))
		cv2.waitKey(100)


def register_points(data_id, args):
	image = cv2.imread('%s/%s_undistorted.jpg' % (args.directory, data_id))
	image = cv2.resize(image, (image.shape[1] / 2, image.shape[0] / 2))
	cv2.namedWindow('image')

	points = []

	def mouse_callback(event, x, y, flags, userdata):
		if event != cv2.EVENT_LBUTTONDOWN:
			return

		points.append((x * 2, y * 2))
		numpy.savetxt('%s/%s_points.csv' % (args.directory, data_id), numpy.array(points))

		color = (0, 255, 0) if len(points) % 2 == 0 else (255, 0, 0)
		cv2.circle(image, (x, y), 5, color, -1)
		cv2.imshow('image', image)
		cv2.imwrite('%s/%s_points.jpg' % (args.directory, data_id), image)

	cv2.setMouseCallback('image', mouse_callback)

	cv2.imshow('image', image)
	cv2.waitKey(0)


def save_pcd(filename, points_3d):
	with open(filename, 'w') as f:
		print >> f, '# .PCD v.7 - Point Cloud Data file format'
		print >> f, 'VERSION .7'
		print >> f, 'FIELDS x y z'
		print >> f, 'SIZE 4 4 4'
		print >> f, 'TYPE F F F'
		print >> f, 'COUNT 1 1 1'
		print >> f, 'WIDTH %d' % points_3d.shape[0]
		print >> f, 'HEIGHT 1'
		print >> f, 'VIEWPOINT 0 0 0 1 0 0 0'
		print >> f, 'POINTS %d' % points_3d.shape[0]
		print >> f, 'DATA ascii'

		for point in points_3d:
			print >> f, '%f %f %f' % tuple(point)


def evaluate_flatness(points_3d):
	mean = numpy.mean(points_3d, axis=0)
	normalized = points_3d - mean

	w, v = numpy.linalg.eig(normalized.T.dot(normalized))

	plane_space = v.dot(normalized.T)
	depths = plane_space[2, :]

	errors = depths - numpy.mean(depths)

	return numpy.sum(numpy.abs(errors)) / points_3d.shape[0]


def reconstruct(params, hand2eye_name, args):
	data_ids = params['data_ids']
	camera_matrix = params['camera_matrix']
	images = [cv2.imread('%s/%s_undistorted.jpg' % (args.directory, x)) for x in data_ids]

	hand2worlds = [numpy.loadtxt('%s/%s_pose.csv' % (args.directory, x)) for x in data_ids]
	world2hands = [numpy.linalg.inv(x) for x in hand2worlds]
	points = [numpy.loadtxt('%s/%s_points.csv' % (args.directory, x)) for x in data_ids]
	points = numpy.rollaxis(numpy.array(points), 0, 2)

	hand2eye = numpy.loadtxt(args.__dict__['hand2eye_' + hand2eye_name])

	hand2eye = numpy.float32(hand2eye).reshape(4, 4)
	hand2eye = hand2eye

	def project(pt, world2hand):
		pt1 = numpy.hstack([pt, [1]])
		xyz = hand2eye.dot(world2hand).dot(pt1)
		uvs = camera_matrix.dot(xyz[:3])
		return uvs[:2] / uvs[-1]

	points_3d = []
	for points_2d in points:
		def error_func(x):
			error = 0
			for i in range(4):
				uv = project(x, world2hands[i])
				error += numpy.sum(numpy.square(points_2d[i] - uv))

			return error

		x0 = [1, 1, 1]
		ret = scipy.optimize.minimize(error_func, x0, method='Nelder-Mead')
		points_3d.append(ret['x'])

	points_3d = numpy.array(points_3d)

	num_total_points = 0
	reproj_error = 0
	for i in range(4):
		for point_2d, point_3d in zip(points[:, i, :], points_3d):
			uv = project(point_3d, world2hands[i])
			num_total_points += 1
			reproj_error += numpy.linalg.norm(uv - point_2d)
			# cv2.circle(images[i], tuple(point_2d.astype(numpy.int32)), 5, (255, 0, 0), -1)
			cv2.circle(images[i], tuple(uv.astype(numpy.int32)), 10, (0, 255, 0), -1)
			# cv2.line(images[i], tuple(point_2d.astype(numpy.int32)), tuple(uv.astype(numpy.int32)), (0, 0, 255), 3)

		cv2.imwrite('%s/%03d_reproj.jpg' % (args.directory, i), images[i])
		cv2.imshow('projected', cv2.resize(images[i], (images[i].shape[1] / 2, images[i].shape[0] / 2)))
		cv2.waitKey(100)
	reproj_error = reproj_error / num_total_points

	# evaluate flatness
	flatness_error = evaluate_flatness(points_3d)

	numpy.savetxt('data/points_3d_%s.csv' % hand2eye_name, points_3d)
	save_pcd('data/points_3d_%s.pcd' % hand2eye_name, points_3d)
	return reproj_error, flatness_error


def main():
	parser = argparse.ArgumentParser(description='3d_reconstruction')
	parser.add_argument('hand2eye_visp')
	parser.add_argument('hand2eye_dq')
	parser.add_argument('hand2eye_graph')
	parser.add_argument('directory')
	parser.add_argument('-c', '--ros_camera_params')

	args = parser.parse_args()

	params = read_params(args)
	undistort(params, args)
	reproj_err_visp, flatness_error_visp = reconstruct(params, 'visp', args)
	reproj_err_dq, flatness_error_dq = reconstruct(params, 'dq', args)
	reproj_err_graph, flatness_error_graph = reconstruct(params, 'graph', args)

	print 'reprj_err_visp', reproj_err_visp, 'reproj_err_dq', reproj_err_dq, 'reproj_err_graph', reproj_err_graph
	print 'flatness_error_visp', flatness_error_visp, 'flatness_error_dq', flatness_error_dq, 'flatness_error_graph', flatness_error_graph


if __name__ == '__main__':
	main()
