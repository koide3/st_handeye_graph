#!/usr/bin/python
# hand-eye calibration simulation with a source-detector camera model
import random
import numpy
import subprocess
import multiprocessing
from matplotlib import pyplot


def simulate(params):
	sim_exe = '../build/handeye_simulation_xray'
	args = ['--seed', str(random.randint(0, 2**30))]
	for arg in params:
		args.append('--' + arg)
		args.append(str(params[arg]))

	p = subprocess.Popen([sim_exe] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output, error = p.communicate()

	lines = output.strip().split('\n')
	result = lines[-1].split(' ')

	print(result)
	# terror_detector, rerror_detector, terror_source
	return result[1::2]


def simulate_ntimes(params, n):
	pool = multiprocessing.Pool(3)

	params_vec = [params for x in range(n)]
	results = pool.map(simulate, params_vec)

	results = numpy.float32(results)
	mean_std = numpy.mean(results, axis=0), numpy.std(results, axis=0)

	print(mean_std[0], mean_std[1])
	return mean_std, results


def plot(noises, results, xlabel, filename):
	pyplot.clf()
	pyplot.subplot(3, 1, 1)
	pyplot.errorbar(noises, results[:, 0], yerr=results[:, 3], label='ours')
	pyplot.xlabel(xlabel)
	pyplot.ylabel('trans_error (detector) [m]')
	pyplot.legend(loc='upper left')
	pyplot.subplot(3, 1, 2)
	pyplot.errorbar(noises, results[:, 1], yerr=results[:, 4], label='ours')
	pyplot.xlabel(xlabel)
	pyplot.ylabel('rot_error (detector) [deg]')
	pyplot.legend(loc='upper left')
	pyplot.subplot(3, 1, 3)
	pyplot.errorbar(noises, results[:, 2], yerr=results[:, 5], label='ours')
	pyplot.xlabel(xlabel)
	pyplot.ylabel('trans_error (source) [deg]')
	pyplot.legend(loc='upper left')
	pyplot.savefig(filename)
	pyplot.pause(1.0)


def main():
	default_params = {
		'visualize': 'false',
		'fx': 300,
		'x_steps': 2,
		'y_steps': 2,
		'z_steps': 1,
		'x_step': 0.5,
		'y_step': 0.5,
		'z_step': 2.0,
		'z_offset': 2.0,
		'hand2detector_trans': 0.3,
		'hand2detector_rot': 15.0,
		'hand2source_trans': 0.0,
		'tnoise': 0.01,
		'rnoise': 0.01,
		'vnoise': 0.1,
		'source_inf_scale': 1.0,
		'visual_inf_scale': 0.01,
		'handpose_inf_scale_trans': 1.0,
		'handpose_inf_scale_rot': 1.0,
		'num_iterations': 1024,
		'solver_name': 'lm_var_cholmod',
		'robust_kernel_handpose': 'Huber',
		'robust_kernel_projection': 'Huber',
		'robust_kernel_source': 'Huber',
		'robust_kernel_handpose_delta': 0.1,
		'robust_kernel_projection_delta': 0.1,
		'robust_kernel_source_delta': 0.1
	}

	# """
	# visual noise
	vnoises = []
	vnoise_results = []
	vnoise_all_results = []
	for vnoise in numpy.arange(0.0, 5.1, 1.0):
		print('vnoise', vnoise)
		params = default_params.copy()
		params['vnoise'] = vnoise

		result, all_result = simulate_ntimes(params, 100)
		vnoises.append(vnoise)
		vnoise_all_results.append(all_result)
		vnoise_results.append(numpy.array(result).flatten())

	numpy.save('data/vnoise_axis_xray.npy', numpy.array(vnoises))
	numpy.save('data/vnoise_results_xray.npy', numpy.array(vnoise_all_results))
	vnoise_results = numpy.array(vnoise_results)
	plot(vnoises, vnoise_results, 'visual noise [pix]', 'data/vnoise_xray.svg')
	# """

	# """
	# trans noise
	tnoises = []
	tnoise_results = []
	tnoise_all_results = []
	for tnoise in numpy.arange(0.0, 0.251, 0.05):
		print('tnoise', tnoise)
		params = default_params.copy()
		params['tnoise'] = tnoise

		result, all_result = simulate_ntimes(params, 100)
		tnoises.append(tnoise)
		tnoise_all_results.append(all_result)
		tnoise_results.append(numpy.array(result).flatten())

	numpy.save('data/tnoise_axis_xray.npy', numpy.array(tnoises))
	numpy.save('data/tnoise_results_xray.npy', numpy.array(tnoise_all_results))
	tnoise_results = numpy.array(tnoise_results)
	plot(tnoises, tnoise_results, 'trans noise [m]', 'data/tnoise_xray.svg')
	# """

	# """
	# rot noise
	rnoises = []
	rnoise_results = []
	rnoise_all_results = []
	for rnoise in numpy.arange(0.0, 10.1, 1.0):
		print('rnoise', rnoise)
		params = default_params.copy()
		params['rnoise'] = rnoise

		result, all_result = simulate_ntimes(params, 100)
		rnoises.append(rnoise)
		rnoise_all_results.append(all_result)
		rnoise_results.append(numpy.array(result).flatten())

	numpy.save('data/rnoise_axis_xray.npy', numpy.array(rnoises))
	numpy.save('data/rnoise_results_xray.npy', numpy.array(rnoise_all_results))
	rnoise_results = numpy.array(rnoise_results)
	plot(rnoises, rnoise_results, 'rotation noise [deg]', 'data/rnoise_xray.svg')
	# """

if __name__ == '__main__':
	main()
