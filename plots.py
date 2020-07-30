import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

from decimal import Decimal


class Experiments(object):
    def __init__(self, directory, reload_logs=False):
        self.directory = directory
        _, self.description = os.path.split(self.directory)
        self.all_experiment_names = os.listdir(self.directory)
        self.all_experiment_names = [e for e in self.all_experiment_names if
                                     os.path.isdir(os.path.join(self.directory, e))]
        self.experiments = [Experiment(os.path.join(self.directory, e),
                                       reload_log=reload_logs) for e in self.all_experiment_names]
        self.grouped_experiments = self.group_experiments()

    def group_experiments(self):
        groups = {}
        for e in self.experiments:
            if e.env_name not in groups:
                groups[e.env_name] = {}
            if e.method not in groups[e.env_name]:
                groups[e.env_name][e.method] = MultipleSeedsExperiment([])
            groups[e.env_name][e.method].add(e)
        return groups


class MultipleSeedsExperiment(object):
    def __init__(self, experiments):
        self.experiments = experiments

    def add(self, experiment):
        self.experiments.append(experiment)

    def get_xs_ys(self, y_series='eprew_recent'):
        list_of_xs, list_of_ys, n_steps_per_update = [], [], []
        for exp in self.experiments:
            list_of_xs.append(exp.timeseries('n_updates'))
            list_of_ys.append(exp.timeseries(y_series))

            n_steps_per_update.append(exp.timeseries('tcount')[-1] / exp.timeseries('n_updates')[-1])
        x, y = group_timeseries(list_of_xs, list_of_ys)
        return x * np.max(n_steps_per_update), y


class Experiment(object):
    log_separator = '------------'

    def __init__(self, directory, reload_log):
        self.directory = directory
        _, self.exp_name = os.path.split(self.directory)
        self.env_name, self.method, self.seed, self.int, self.ext = self.parse_name(name=self.exp_name)
        self.log = self.load_log(regenerate=reload_log)
        self._timeseries = {}

    def parse_name(self, name):
        env, seed, method, int, ext = name.split('_')
        int = Decimal(int.split('-')[1])
        ext = Decimal(ext.split('-')[1])
        if method == 'none':
            method = 'rf'
        if ext == 1 and int == 0:
            method = 'none'
        method = f"{method}_{int}_{ext}"
        return env, method, seed, int, ext

    def load_log(self, regenerate=False):
        pickle_of_log = os.path.join(self.directory, 'log.pickle')
        if os.path.exists(pickle_of_log) and not regenerate:
            with open(pickle_of_log, 'rb') as f:
                print('loading log ', pickle_of_log)
                log = pickle.load(f)
        else:
            print('parsing log {}'.format(self.directory))
            log_file = os.path.join(self.directory, 'log.txt')

            log = self.parse_log_from_file(log_file)
            with open(pickle_of_log, 'wb') as f:
                pickle.dump(log, f, protocol=-1)
        return log

    def parse_log_from_file(self, log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines if l.startswith('|') or l.startswith(Experiment.log_separator)]
        chunks = []
        current_chunk = {}
        for l in lines:
            if l.startswith(Experiment.log_separator):
                if current_chunk != {}:  # end of chunk
                    chunks.append(current_chunk)
                current_chunk = {}
            else:
                name, value = [p.strip() for p in l.split('|') if p.strip() != '']
                try:
                    if value != 'nan':
                        value = eval(value)
                    else:
                        value = np.nan
                except:
                    print(name, value, "failed")
                current_chunk[name] = value
        return chunks

    def timeseries(self, name):
        assert self.log is not None
        if self._timeseries.get(name, None) is None:
            self._timeseries[name] = [entry.get(name, 0.) for entry in self.log]
        return self._timeseries[name]

    def __str__(self):
        return self.exp_name


def group_timeseries(list_of_xs, list_of_ys):
    grouped_values = {}
    for xs, ys in zip(list_of_xs, list_of_ys):
        for x, y in zip(xs, ys):
            if x not in grouped_values:
                grouped_values[x] = []
            if y is None:
                y = np.nan
            grouped_values[x].append(y)
    # take the intersection
    xs = []
    ys = []
    for x in grouped_values:
        v = grouped_values[x]
        assert len(v) <= len(list_of_ys)
        if len(v) == len(list_of_ys):
            xs.append(x)
            ys.append(v)
    return np.asarray(xs), np.asarray(ys)


def smooth(x, extent):
    from scipy import signal
    if extent is not 0:
        return signal.savgol_filter(np.nan_to_num(x), extent, 3)
    else:
        return np.nan_to_num(x)


class AxesWithPlots(object):
    def __init__(self, ax):
        self.ax = ax
        self.max_xs = []
        self.lines = []

    def add_std_plot(self, x, y, color, label, smoothen=51, alpha=1., std_alpha=0.3, frames_per_timestep=4, clip=None):
        x *= frames_per_timestep
        x /= 1e6
        if smoothen != 0:
            y = np.stack([smooth(y[:, i], extent=smoothen) for i in range(y.shape[1])], 1)
            if clip is not None:
                y = np.clip(y, clip[0], clip[1])
        mean_y = np.mean(y, 1)

        self.lines.extend(self.ax.plot(x, mean_y, color=color, label=label, alpha=alpha))
        if y.shape[1] > 1:
            std_y = np.std(y, 1, ddof=1)
            upper = mean_y + std_y / np.sqrt(y.shape[1])
            lower = mean_y - std_y / np.sqrt(y.shape[1])
            self.ax.fill_between(x, upper, lower, color=color, alpha=std_alpha, lw=0.)

        self.max_xs.append(x[-1])

    def finish_up(self, title, fontsize=8, xlim=300, tight_y=True):
        self.ax.set_xlim([0, xlim])
        self.ax.set_title(title, fontsize=fontsize)
        self.ax.autoscale(enable=True, axis='y', tight=tight_y)


def label(method):
    method, int, ext = method.split('_')
    if method == 'none':
        return labels[method]
    else:
        return f"{labels[method]}: INT={int},EXT={ext}"


def color(method):
    method, int, ext = method.split('_')
    int = Decimal(int)
    if method == 'none':
        return 'green'
    if method == 'idf':
        if int == 1:
            return 'red'
        return 'orange'
    if method == 'rf':
        if int == 1:
            return 'blue'
        return 'purple'


labels = {
    'rf': 'Random Features',
    'idf': 'Inverse Dynamic Features',
    'none': 'Extrinsic Only'
}


def generate_three_seed_graph(experiment, y_series='eprew_recent', smoothen=51, alpha=1.0, xlim=100):
    num_envs = len(experiment.grouped_experiments)
    print(num_envs)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12, 6))
    all_axes = []
    envs = ['Freeway']
    # envs = ['Breakout', 'MontezumaRevenge', 'Freeway', 'Frostbite']

    for env, ax in zip(envs, np.ravel(axes)):
        ax.tick_params(labelsize=7, pad=0.001)
        ax_with_plots = AxesWithPlots(ax)
        all_axes.append(ax_with_plots)
        for method in experiment.grouped_experiments[env]:
            if method != 'extrew':
                print("generating graph ", env, method)
                xs, ys = experiment.grouped_experiments[env][method].get_xs_ys(y_series)
                ax_with_plots.add_std_plot(xs, ys, color=color(method), label=label(method), smoothen=smoothen,
                                           alpha=alpha)
        ax_with_plots.finish_up(title=env, xlim=xlim)

    plt.tight_layout(pad=1.5, w_pad=-0.3, h_pad=0.3, rect=[0.0, 0., 1, 1])
    # fig.legend(handles=all_axes[0].lines, borderaxespad=0., fontsize=10, loc=(0.6, 0.2), ncol=1)
    fig.text(0.5, 0.01, 'Frames (millions)', ha='center')
    fig.text(0.001, 0.5, 'Extrinsic Reward per Episode', va='center', rotation='vertical')

    save_filename = os.path.join(results_folder, '{}_{}.png'.format(envs[0], y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=300)
    plt.close()


def main():
    experiment = Experiments(os.path.join(results_folder), reload_logs=False)
    generate_three_seed_graph(experiment, 'retmean', xlim=100)
    generate_three_seed_graph(experiment, 'rew_mean', xlim=100)
#    generate_three_seed_graph(experiment, 'best_ext_ret', smoothen=False, xlim=100)
    print('parsed')


if __name__ == '__main__':
    results_folder = '../../test/FreewayFINAL'
    os.chdir(results_folder)
    main()
