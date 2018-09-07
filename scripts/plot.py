#!/usr/bin/python
import numpy
from matplotlib import pyplot


def plot(ticks, results, xlabel, col, filename):
	terror_min = -0.05
	terror_max = 0.5
	rerror_min = -0.5
	rerror_max = 15.0

	means = []
	stds = []
	for result in results:
		mean = numpy.mean(result, axis=0)
		std = numpy.std(result, axis=0)

		means.append(mean)
		stds.append(std)

	means = numpy.array(means)
	stds = numpy.array(stds)

	pyplot.subplot(2, 3, col)
	pyplot.errorbar(ticks, means[:, 0], yerr=stds[:, 0], label='ours')
	pyplot.errorbar(ticks, means[:, 2], yerr=stds[:, 2], label='tsai')
	pyplot.errorbar(ticks, means[:, 4], yerr=stds[:, 2], label='dual_quat')
	pyplot.ylim(ymin=terror_min, ymax=terror_max)
	pyplot.subplot(2, 3, col+3)
	pyplot.errorbar(ticks, means[:, 1], yerr=stds[:, 1], label='ours')
	pyplot.errorbar(ticks, means[:, 3], yerr=stds[:, 3], label='tsai')
	pyplot.errorbar(ticks, means[:, 5], yerr=stds[:, 5], label='dual_quat')
	pyplot.xlabel(xlabel)
	pyplot.ylim(ymin=rerror_min, ymax=rerror_max)


def plot_pinhole():
	fig = pyplot.figure(figsize=(12, 4))
	pyplot.subplot(2, 3, 1)
	pyplot.ylabel('Translation\nerror [m]')
	pyplot.subplot(2, 3, 4)
	pyplot.ylabel('Rotation\nerror [deg]')

	vnoises = numpy.load('data/vnoise_axis.npy')
	vnoise_results = numpy.load('data/vnoise_results.npy')
	plot(vnoises, vnoise_results, 'Visual noise [pix]', 1, 'vnoise.svg')

	tnoises = numpy.load('data/tnoise_axis.npy')
	tnoise_results = numpy.load('data/tnoise_results.npy')
	plot(tnoises, tnoise_results, 'Translation noise [m]', 2, 'tnoise.svg')

	rnoises = numpy.load('data/rnoise_axis.npy')
	rnoise_results = numpy.load('data/rnoise_results.npy')
	plot(rnoises, rnoise_results, 'Rotation noise [deg]', 3, 'rnoise.svg')

	ax = pyplot.subplot(2, 3, 1)
	handles, labels = ax.get_legend_handles_labels()

	pyplot.tight_layout()
	pyplot.subplots_adjust(bottom=0.25)
	fig.legend(handles, labels, loc='lower center', ncol=3)

	pyplot.savefig('data/pinhole_result.svg')
	pyplot.show()


def plot_(ticks, results, col, xlabel):
	terror_min = 0.0
	terror_max = 0.1
	rerror_min = 0.0
	rerror_max = 2.5

	means = []
	stds = []
	for result in results:
		mean = numpy.mean(result, axis=0)
		std = numpy.std(result, axis=0)

		means.append(mean)
		stds.append(std)

	means = numpy.array(means)
	stds = numpy.array(stds)

	pyplot.subplot(3, 3, col)
	pyplot.errorbar(ticks, means[:, 0], yerr=stds[:, 0])
	pyplot.ylim(ymin=terror_min, ymax=terror_max)
	pyplot.subplot(3, 3, col+3)
	pyplot.errorbar(ticks, means[:, 1], yerr=stds[:, 1])
	pyplot.ylim(ymin=rerror_min, ymax=rerror_max)
	pyplot.subplot(3, 3, col+6)
	pyplot.errorbar(ticks, means[:, 2], yerr=stds[:, 2])
	pyplot.ylim(ymin=terror_min, ymax=terror_max)
	pyplot.xlabel(xlabel)


def plot_xray():
	pyplot.figure(figsize=(12, 5))
	pyplot.subplot(3, 3, 1)
	pyplot.ylabel('Detector\ntranslation\nerror [m]')
	pyplot.subplot(3, 3, 4)
	pyplot.ylabel('Detector\nrotation\nerror [deg]')
	pyplot.subplot(3, 3, 7)
	pyplot.ylabel('Source\ntranslation\nerror [m]')

	vnoises = numpy.load('data/vnoise_axis_xray.npy')
	vnoise_results = numpy.load('data/vnoise_results_xray.npy')
	plot_(vnoises, vnoise_results, 1, 'Visual noise [pix]')

	tnoises = numpy.load('data/tnoise_axis_xray.npy')
	tnoise_results = numpy.load('data/tnoise_results_xray.npy')
	plot_(tnoises, tnoise_results, 2, 'Translation noise [m]')

	rnoises = numpy.load('data/rnoise_axis_xray.npy')
	rnoise_results = numpy.load('data/rnoise_results_xray.npy')
	plot_(rnoises, rnoise_results, 3, 'Rotation noise [deg]')

	pyplot.tight_layout()
	pyplot.savefig('data/xray_result.svg')
	pyplot.show()


if __name__ == '__main__':
	plot_pinhole()
	plot_xray()
