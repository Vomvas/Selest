""" Classes to draw curves on subplots """
from collections import deque
import copy


__all__ = ['STANDARD_COLORS', 'CurveData', 'BarData', 'ChartData', 'LineChart', 'MarkerChart', 'BarGroup', "BarChart"]


STANDARD_COLORS = ['#ff7700', 'r', 'b', 'g', 'c', 'm', 'y', 'k',
                   '#9933ff', '#00cc66', '#770000', '#666699',
                   '#cc9900', '#993333', '#669999', '#7777cc']


class CurveData(object):
    """
    Stores X and Y axis data for a curve to be plotted.
    Optionally the label can be given to generate the chart legend.
    """
    def __init__(self, x, y, label=None):
        self.x = x
        self.y = y
        self.label = label


class BarData(object):
    """
    Stores X and Y axis data for a curve to be plotted.
    Optionally the label can be given to generate the chart legend.
    """
    def __init__(self, x, height, width=0.01, label=None):
        self.x = x
        self.height = height
        self.width = width
        self.label = label


class BarGroup(object):
    """ Group of bar plots or group of group of bar plots """
    def __init__(self, bars, x=None, label=None):
        self.bars = bars
        self.x = x
        self.label = label


class ChartData(object):
    """
    Stores the chart data such as title, X and Y axis labels, and curves' data.
    """
    def __init__(self, curves=None, bars=None, markers=None, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None,
                 colors=None, linestyles=None, linewidths=None, linemarkers=None, fillstyles=None, bargroups=None,
                 options=None):
        self.title = title if title is not None else ''
        self.xlabel = xlabel if xlabel is not None else ''
        self.ylabel = ylabel if ylabel is not None else ''
        self.xlim = xlim
        self.ylim = ylim
        self.colors = colors
        self.linestyles = linestyles
        self.linewidths = linewidths
        self.linemarkers = linemarkers
        self.options = options
        self.curves = curves if curves is not None else []
        self.bars = bars if bars is not None else []
        self.bargroups = bargroups if bargroups is not None else []
        self.markers = markers if markers is not None else []
        self.fillstyles = fillstyles

    def set_chart_properties(self, subplot):
        """ Sets up properties such as title and axes labels """
        subplot.set_title(self.title)
        subplot.set_xlabel(self.xlabel)
        subplot.set_ylabel(self.ylabel)
        subplot.set_xlim(self.xlim)
        subplot.set_ylim(self.ylim)
        subplot.grid(b=True)
        subplot.minorticks_on()
        subplot.tick_params(axis='both', which='major', labelsize=20)
        subplot.tick_params(axis='both', which='minor', labelsize=16)

        if self.options:
            if self.options.get("title"):
                subplot.set_title(self.options.get("title"), fontsize=self.options.get("titlesize", 16))
            if self.options.get("fontsize"):
                subplot.xaxis.label.set_fontsize(int(self.options.get("fontsize")))
                subplot.yaxis.label.set_fontsize(int(self.options.get("fontsize")))
            if self.options.get("x_ticklabels"):
                subplot.set_xticks(self.options.get("x_ticks"))
                subplot.set_xticklabels(self.options.get("x_ticklabels"))
            if self.options.get("y_ticklabels"):
                subplot.set_yticks(self.options.get("y_ticks"))
                subplot.set_yticklabels(self.options.get("y_ticklabels"))

    def set_chart_format(self):
        """ Get properties regarding chart appearance """
        self.colors = deque(STANDARD_COLORS if self.colors is None else self.colors)
        self.linestyles = deque(['-'] if self.linestyles is None else self.linestyles)
        self.linewidths = deque([2] if self.linewidths is None else self.linewidths)
        self.linemarkers = deque([''] if self.linemarkers is None else self.linemarkers)
        self.fillstyles = deque(['full'] if self.fillstyles is None else self.fillstyles)


class LineChart(object):
    """
    Provides means to easily draw a line chart consisting of multiple curves.
    """

    @staticmethod
    def plot(subplot, chart_data, options=None):
        """ Plot lines on a subplot """

        chart_data.set_chart_format()

        legend_handles = []
        for curve in chart_data.curves:
            subplot.plot(curve.x, curve.y, color=chart_data.colors[0], linestyle=chart_data.linestyles[0],
                         linewidth=chart_data.linewidths[0],
                         marker=chart_data.linemarkers[0], label=curve.label)

            chart_data.colors.rotate(-1)
            chart_data.linestyles.rotate(-1)
            chart_data.linewidths.rotate(-1)
            chart_data.linemarkers.rotate(-1)
            chart_data.fillstyles.rotate(-1)

        if chart_data.options is not None:
            if options is not None:
                options.update(chart_data.options)
            else:
                options = chart_data.options

        chart_data.set_chart_properties(subplot)

        if options is None:
            return

        if 'yticks' in options:
            yticks_options = copy.copy(options['yticks'])
            yticks = yticks_options.pop('ticks', None)
            yticks_labels = yticks_options.pop('labels', None)
            if yticks is not None:
                subplot.set_yticks(yticks)
            if yticks_labels is not None:
                subplot.set_yticklabels(yticks_labels, **yticks_options)

        if 'xticks' in options:
            xticks_options = copy.copy(options['xticks'])
            xticks = xticks_options.pop('ticks', None)
            xticks_labels = xticks_options.pop('labels', None)
            if xticks is not None:
                subplot.set_yticks(xticks)
            if xticks_labels is not None:
                subplot.set_yticklabels(xticks_labels, **xticks_options)

        if 'legend' in options:
            subplot.legend(**options['legend'])
        else:
            subplot.legend()


class BarChart(object):
    """
    Provides means to easily draw a bar chart consisting of multiple bars.
    """

    @staticmethod
    def plot(subplot, chart_data, options=None):
        """ Plot bars on a subplot """

        chart_data.set_chart_format()

        legend_handles = []
        group_labels = [g.label for g in chart_data.bargroups]
        x_values = [(x+0.3)/5 for x in range(len(group_labels))]
        if not chart_data.bargroups:
            bg = BarGroup(chart_data.bars)
            chart_data.bargroups = [bg]
        for bargroup in chart_data.bargroups:
            bargroup.x = x_values.pop(0)
            for bar in bargroup.bars:
                bar.x = bargroup.x - (len(bargroup.bars) - 1) * bar.width / 2 + bargroup.bars.index(bar) * bar.width
                t = subplot.bar(bar.x, bar.height, bar.width, color=chart_data.colors[0],
                            linestyle=chart_data.linestyles[0], label=bar.label)
                legend_handles.append(t)
                chart_data.colors.rotate(-1)
            chart_data.colors.rotate(len(bargroup.bars))

        if chart_data.options is not None:
            if options is not None:
                options.update(chart_data.options)
            else:
                options = chart_data.options

        chart_data.set_chart_properties(subplot)
        subplot.set_xticks([x.x for x in chart_data.bargroups])
        subplot.set_xticklabels([x.label for x in chart_data.bargroups])

        if 'legend' in options:
            subplot.legend(**options['legend'], fontsize=40)
        else:
            same_labels = [x.label for x in chart_data.bargroups[0].bars]
            subplot.legend(legend_handles, same_labels, fontsize=40)


class MarkerChart(object):
    """ Add the chart markers """
    @staticmethod
    def plot(subplot, chart_data, options=None):
        """ Helper function """
        for marker in chart_data.markers:
            subplot.add_patch(marker)
