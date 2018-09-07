#!/usr/bin/python
# hand-eye calibration simulation under visual and hand pose noises
import random
import numpy
import subprocess
import multiprocessing
from matplotlib import pyplot


def simulate(params):
	sim_exe = '../build/handeye_simulation'
	args = ['--seed', str(random.randint(0, 2**30))]
	for arg in params:
		args.append('--' + arg)
		args.append(str(params[arg]))

	p = subprocess.Popen([sim_exe] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	p.wait()

	lines = p.stdout.readlines()
	result = lines[-1].split(' ')

	# terror_graph, rerror_graph, terror_visp, rerror_visp, terror_dq, rerror_dq
	return result[1::2]


def simulate_ntimes(params, n):
	pool = multiprocessing.Pool(None)

	params_vec = [params for x in range(n)]
	results = pool.map(simulate, params_vec)

	results = numpy.float32(results)
	mean_std = numpy.mean(results, axis=0), numpy.std(results, axis=0)

	print mean_std[0], mean_std[1]
	return mean_std, results


def plot(noises, results, xlabel, filename):
	pyplot.clf()
	pyplot.subplot(2, 1, 1)
	pyplot.errorbar(noises, results[:, 0], yerr=results[:, 6], label='ours')
	pyplot.errorbar(noises, results[:, 2], yerr=results[:, 8], label='tsai')
	pyplot.errorbar(noises, results[:, 4], yerr=results[:, 10], label='dual_quat')
	pyplot.xlabel(xlabel)
	pyplot.ylabel('trans_error [m]')
	pyplot.legend(loc='upper left')
	pyplot.subplot(2, 1, 2)
	pyplot.errorbar(noises, results[:, 1], yerr=results[:, 7], label='ours')
	pyplot.errorbar(noises, results[:, 3], yerr=results[:, 9], label='tsai')
	pyplot.errorbar(noises, results[:, 5], yerr=results[:, 11], label='dual_quat')
	pyplot.xlabel(xlabel)
	pyplot.ylabel('rot_error [deg]')
	pyplot.legend(loc='upper left')
	pyplot.savefig(filename)
	pyplot.pause(1.0)


def main():
	random.seed(10)

	default_params = {
		'visualize': 'false',
		'x_steps': 1,
		'y_steps': 1,
		'z_steps': 1,
		'x_step': 0.5,
		'y_step': 0.5,
		'z_step': 0.25,
		'z_offset': 0.75,
		'hand2eye_trans': 0.3,
		'hand2eye_rot': 90.0,
		'tnoise': 0.01,
		'rnoise': 0.01,
		'vnoise': 0.1,
		'use_init_guess': 'true',
		'visual_inf_scale': 1.0,
		'handpose_inf_scale_trans': 1e-3,
		'handpose_inf_scale_rot': 1.0,
		'num_iterations': 1024,
		'solver_name': 'lm_var_cholmod',
		'robust_kernel_handpose': 'Huber',
		'robust_kernel_projection': 'Huber',
		'robust_kernel_handpose_delta': 0.01,
		'robust_kernel_projection_delta': 1.0
	}

	# """
	# visual noise
	vnoises = []
	vnoise_results = []
	vnoise_all_results = []
	for vnoise in numpy.arange(0.0, 5.1, 1.0):
		print 'vnoise', vnoise
		params = default_params.copy()
		params['vnoise'] = vnoise

		result, all_results = simulate_ntimes(params, 100)
		vnoises.append(vnoise)
		vnoise_all_results.append(all_results)
		vnoise_results.append(numpy.array(result).flatten())

	numpy.save('data/vnoise_axis.npy', numpy.array(vnoises))
	numpy.save('data/vnoise_results.npy', numpy.array(vnoise_all_results))
	vnoise_results = numpy.array(vnoise_results)
	plot(vnoises, vnoise_results, 'visual noise [pix]', 'data/vnoise.svg')
	# """

	# """
	# trans noise
	tnoises = []
	tnoise_results = []
	tnoise_all_results = []
	for tnoise in numpy.arange(0.0, 0.251, 0.05):
		print 'tnoise', tnoise
		params = default_params.copy()
		params['tnoise'] = tnoise

		result, all_results = simulate_ntimes(params, 100)
		tnoises.append(tnoise)
		tnoise_all_results.append(all_results)
		tnoise_results.append(numpy.array(result).flatten())

	numpy.save('data/tnoise_axis.npy', numpy.array(tnoises))
	numpy.save('data/tnoise_results.npy', numpy.array(tnoise_all_results))
	tnoise_results = numpy.array(tnoise_results)
	plot(tnoises, tnoise_results, 'trans noise [m]', 'data/tnoise.svg')
	# """

	# """
	# rot noise
	rnoises = []
	rnoise_results = []
	rnoise_all_results = []
	for rnoise in numpy.arange(0.0, 10.1, 1.0):
		print 'rnoise', rnoise
		params = default_params.copy()
		params['rnoise'] = rnoise

		result, all_results = simulate_ntimes(params, 100)
		rnoises.append(rnoise)
		rnoise_all_results.append(all_results)
		rnoise_results.append(numpy.array(result).flatten())

	numpy.save('data/rnoise_axis.npy', numpy.array(rnoises))
	numpy.save('data/rnoise_results.npy', numpy.array(rnoise_all_results))
	rnoise_results = numpy.array(rnoise_results)
	plot(rnoises, rnoise_results, 'rotation noise [deg]', 'data/rnoise.svg')
	# """

if __name__ == '__main__':
	main()
