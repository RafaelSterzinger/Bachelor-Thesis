import os
import matplotlib.pyplot as plt
import numpy as np
import pickle


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
        self.env_name, self.method, self.seed = self.parse_name(name=self.exp_name)
        self.log = self.load_log(regenerate=reload_log)
        # self.exp_details = self.load_exp_info()
        self._timeseries = {}

    def parse_name(self, name):
        if name.startswith('mariogen'):
            return self.parse_name_mariogen(name)
        elif name == 'mz8gpu':
            return 'MontezumaRevenge', 'idf', 0
        elif name == 'mario16gpurf-00':
            return 'Mario', 'rf_large', 0
        elif name == 'mario1gpurf':
            return 'Mario', 'rf_small', 0
        else:
            if '_' in self.exp_name:
                return self.parse_name_yura_convention(name)
            else:
                return self.parse_name_rafael_convention(name)

    def parse_name_yura_convention(self, name):
        env, method, seed, *extra = name.split('_')
        if method == 'randfeat':
            method = 'rf'
        if extra != []:
            method += '_' + '_'.join(extra)
        return env, method, seed

    def parse_name_rafael_convention(self, name):
        envs = {'mz': 'MontezumaRevenge',
                'mario': 'Mario',
                }
        methods = ['vaesph', 'pix2pixbn', 'pix2pix', 'idfbnnontrainable', 'idfbntrainable', 'idf', 'rf', 'vaenonsph',
                   'randfeatbn', 'ext']  # the order matters
        method = [m for m in methods if m in name]
        method = method[0]
        seed = int(name[-1]) - 1
        env = name[:-(1 + len(method))]
        env = envs[env]
        return env, method, seed

    def parse_name_mariogen(self, name):
        name = name[len('mariogen'):]
        env = 'MarioGen'
        if name.startswith('fixed'):
            extra_method = '_fixed'
            name = name[len('fixed'):]
        else:
            extra_method = ''
        method = name[:len('1to1')]
        name = name[len('1to1'):]
        assert name.startswith('idf') or name.startswith('rf'), self.directory
        if name.startswith('idf'):
            feat_method = 'idf'
        elif name.startswith('rf'):
            feat_method = 'rf'
        method += '_{}'.format(feat_method)
        method += extra_method
        name = name[len(feat_method):]
        seed = int(name) - 1
        return env, method, seed

    def load_exp_info(self):
        directory = os.path.join(self.directory, 'exp_details.pkl')
        if os.path.exists(directory):
            with open(directory, 'rb') as f:
                try:
                    exp_details = pickle.load(f)
                    if not exp_details:
                        exp_details = None
                except:
                    exp_details = None
        else:
            exp_details = None
        return exp_details

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
        return signal.savgol_filter(np.nan_to_num(x), extent,
                                    3)  # TODO: change the smoothing back to 101 when all the datapoints are in
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

    def finish_up(self, title, fontsize=8, xlim=400, tight_y=True):
        self.ax.set_xlim([0, min(xlim, max(*self.max_xs))])
        self.ax.set_title(title, fontsize=fontsize)
        self.ax.autoscale(enable=True, axis='y', tight=tight_y)


colors = {'idf': 'orange',
          'idf_3epochs': 'orange',
          'idf_6epochs': 'red',
          'vaesph': 'cornflowerblue',
          'rf': 'seagreen',
          'rf_6epochs': 'seagreen',
          'rf_3epochs': 'black',
          'extrew': 'gray',
          'ext': 'gray',
          'pix2pix': 'lightcoral'
          }
labels = {'idf': 'Inverse Dynamics\nfeatures',
          'idf_3epochs': 'Inverse Dynamics features 3 ep',
          'idf_6epochs': 'Inverse Dynamics features',
          'pix2pix': 'Pixels',
          'rf': 'Random CNN\nfeatures',
          'rf_3epochs': 'Random CNN features 3 ep',
          'rf_6epochs': 'Random CNN features',
          'extrew': 'Extrinsic rewards',
          'ext': 'Extrinsic rewards',
          'vaesph': 'VAE features'
          }


def generate_three_seed_graphs(three_seed_exps, y_series='eprew_recent', smoothen=51, alpha=1.0):
    num_envs = len(three_seed_exps.grouped_experiments)
    print(num_envs)
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, figsize=(12, 6))
    all_axes = []
    envs = [env for env in three_seed_exps.grouped_experiments if env not in ['Unity-Sparse']]
    envs = ['BeamRider', 'Breakout', 'MontezumaRevenge', 'Pong', 'Mario', 'Qbert', 'Riverraid', 'Seaquest',
            'SpaceInvaders']
    # envs.remove('Mario')
    for env, ax in zip(envs, np.ravel(axes)):
        # import ipdb; ipdb.set_trace()
        ax.tick_params(labelsize=7, pad=0.001)
        ax_with_plots = AxesWithPlots(ax)
        all_axes.append(ax_with_plots)
        for method in three_seed_exps.grouped_experiments[env]:
            if method != 'extrew':
                print("generating graph ", env, method)
                xs, ys = three_seed_exps.grouped_experiments[env][method].get_xs_ys(y_series)
                ax_with_plots.add_std_plot(xs, ys, color=colors[method], label=labels[method], smoothen=smoothen,
                                           alpha=alpha)
        ax_with_plots.finish_up(title=env)
    axes[1, 4].set_visible(False)
    # fig.set_xlabel('Number of training steps (in millions)', fontsize=14)
    # fig.set_ylabel('Extrinsic Rewards per Episode', fontsize=14)
    plt.tight_layout(pad=1.5, w_pad=-0.3, h_pad=0.3, rect=[0.0, 0., 1, 1])
    # plt.tight_layout()
    fig.legend(handles=all_axes[0].lines, borderaxespad=0., fontsize=10, loc=(0.82, 0.2), ncol=1)
    fig.text(0.5, 0.01, 'Frames (millions)', ha='center')
    fig.text(0.007, 0.5, 'Extrinsic Reward per Episode', va='center', rotation='vertical')
    save_filename = os.path.join(results_folder, 'three_seeds_{}.png'.format(y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=300)
    plt.close()


def generate_unitytv_graphs(unity_exps, y_series='eprew_recent', smoothen=51, alpha=1.0):
    colors = {
        'rf-TV0': 'seagreen',
        'rf-TV2': 'cornflowerblue',
        'idf-TV0': 'orange',
        'idf-TV2': 'lightcoral'
    }
    labels = {
        'rf-TV0': 'RF with TV off',
        'rf-TV2': 'RF with TV on',
        'idf-TV0': 'IDF with TV off',
        'idf-TV2': 'IDF with TV on'
    }
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax = axes
    envs = [env for env in unity_exps.grouped_experiments]
    for env in envs:
        # import ipdb; ipdb.set_trace()
        ax.tick_params(labelsize=12, pad=0.001)
        ax_with_plots = AxesWithPlots(ax)
        for method in unity_exps.grouped_experiments[env]:
            print("generating graph ", env, method)
            xs, ys = unity_exps.grouped_experiments[env][method].get_xs_ys(y_series)
            method = method + '-' + env[-3:]
            ax_with_plots.add_std_plot(xs, ys, color=colors[method], label=labels[method], smoothen=smoothen,
                                       alpha=alpha, clip=[0., 1.])
        ax_with_plots.finish_up(title='Unity maze', fontsize=18)
    ax.set_xlim([0, 8.5])
    # fig.set_xlabel('Number of training steps (in millions)', fontsize=14)
    # fig.set_ylabel('Extrinsic Rewards per Episode', fontsize=14)
    plt.tight_layout(pad=1.5, w_pad=-0.3, h_pad=0.3, rect=[0.01, 0.01, 1, 0.925])
    # plt.tight_layout()
    fig.legend(borderaxespad=0., fontsize=12, loc='upper center', ncol=2)
    fig.text(0.5, 0.01, 'Frames (millions)', ha='center', fontsize=18)
    fig.text(0.007, 0.5, 'Extrinsic Reward per Episode', va='center', rotation='vertical', fontsize=18)
    save_filename = os.path.join(results_folder, 'three_seeds_unitytv_{}.png'.format(y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=300)
    plt.close()


def main():
    # Code for Figure-2
    three_seed_exps = Experiments(os.path.join(results_folder, 'three_seed_main_envs'), reload_logs=False)
    generate_three_seed_graphs(three_seed_exps, 'eprew_recent')
    generate_three_seed_graphs(three_seed_exps, 'recent_best_ext_ret', smoothen=False, alpha=0.7)
    generate_three_seed_graphs(three_seed_exps, 'best_ext_ret', smoothen=False)
    generate_three_seed_graphs(three_seed_exps, 'eplen')


if __name__ == '__main__':
    results_folder = os.chdir("/tmp")
    results_folder = '/tmp'
    main()
