""" Plotter for benchmarks """
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec


class Plotter(object):
    """ Helper class for plotting benchmarks """

    @staticmethod
    def new_figure(figshape=(1, 1), options=None):
        """ Return a new set of figure - subplot """
        if options is None:
            options = {}
        rows, cols = figshape
        return plt.subplots(rows, cols, **options)

    @staticmethod
    def show_plot():
        plt.show()

    @staticmethod
    def plot(benchmark, subplot=None, outfile=None, figdir=None, options=None):
        """
        If subplot is given, the chart is drawed to the subplot.
        Otherwise, the chart is drawed and saved to a file.

        :param benchmark: benchmark object
        :param subplot: subplot object
        :param outfile: output filename
        :param figdir: output directory
        :param options: chart options
        """
        if subplot is not None:
            Plotter._plot(benchmark, subplot, options=options)
        else:
            Plotter._save_to_file(benchmark, outfile=outfile, figdir=figdir, options=options)

    @staticmethod
    def _save_to_file(benchmark, outfile=None, figdir=None, options=None):

        # create a new figure object
        if options and options.get('figwidth', None) and options.get('figheight', None):
            plt.figure(figsize=(options['figwidth'], options['figheight']))
        else:
            plt.figure()

        gridspec = GridSpec(1, 1)
        subplot = plt.subplot(gridspec[0, 0])

        Plotter._plot(benchmark, subplot, options)
        if benchmark.get_plot_config().get("style", None) == "bars":
            plt.tight_layout()
        # plt.tight_layout()
        filename = Plotter.gen_fig_filename(benchmark, options=options, figdir=figdir, outfile=outfile)
        plt.savefig(filename, format='pdf', dpi=300)

    @staticmethod
    def _plot(benchmark, subplot, options=None):
        benchmark.plot_benchmark(subplot, options=options)

    @staticmethod
    def gen_fig_filename(benchmark, figdir=None, outfile=None, options=None):
        """ Generate filename for benchmark plot """
        if figdir and outfile:
            return os.path.join(figdir, outfile)
        elif outfile:
            return outfile
        else:
            suffix = options.get("separate") if options.get("separate") else None
            figdir = figdir or benchmark.outputs_dir
            return os.path.join(figdir, benchmark.get_figure_filename(suffix=suffix))
