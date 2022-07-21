import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def classification_timecourse(classification_vals, path_to_ev_files, fig_save_name, tr=.72):
    """

        :param classification_vals:
        :param path_to_ev_files:
        :param fig_save_name:
        :param tr:
        :return:
        """

    output = np.argmax(classification_vals, axis=1) + 1
    n_events = len(path_to_ev_files)
    fig, axes = plt.subplots(len(path_to_ev_files), 1, figsize=(12, 8), sharex=True, sharey=True)

    x = np.arange(0, classification_vals.shape[0])

    for ev in range(n_events):

        event_tc = np.zeros(len(x))
        classifier_decision = event_tc.copy()
        classifier_decision[output == ev + 1] = 1

        ev_info = pd.read_table(path_to_ev_files[ev], header=None)
        ev_start = np.array(np.ceil(ev_info[0] / tr)).astype(int)
        ev_end = np.array(ev_start + np.ceil(ev_info[1] / tr)).astype(int)

        for i in range(len(ev_start)):
            if 'rest' in path_to_ev_files[ev]:
                indices = np.arange(ev_start[i], ev_end[i])
            else:
                indices = np.arange(ev_start[i], ev_end[i]) + 4
            event_tc[indices] = 1

        axes[ev].plot(x, event_tc, label='design', linewidth=5, color='b')
        axes[ev].plot(x, np.transpose(classification_vals[:, ev]), label='confidence', linestyle='--',
                      linewidth=3, color='r')
        axes[ev].plot(x, classifier_decision, linestyle=':', linewidth=1, color='k', label='argmax')

        # horizontal line to indicate chance level
        axes[ev].axhline(y=1 / n_events, linewidth=2, color='gray', linestyle='-',
                         label='chance')
        axes[ev].set_title(str(os.path.split(path_to_ev_files[ev])[-1]).split('_RL.txt')[0])
        axes[ev].spines['right'].set_visible(False)
        axes[ev].spines['top'].set_visible(False)

        if ev == len(path_to_ev_files) - 1:
            axes[ev].set_xlabel('volume')
            axes[ev].legend(loc='upper right', fontsize=10)
        axes[ev].set_ylabel('confidence')

    fig.tight_layout()

    plt.savefig(fig_save_name, facecolor=fig.get_facecolor(), transparent=True)
    # plt.close(fig)


def confusion_matrix(real, predicted, classes, fig_save_name=None, normalize=False, doplot=True, ax=None):
    """
    function to plot a confusion matrix

    :param real:            vector of real labels
    :param predicted:       vector of predicted labels
    :param classes:         list of class names
    :param fig_save_name:   output path + name of the figure
    :param normalize:       bool normalize to values between 0-1 [True | (default)False]
    :param doplot:          bool to show the plot [(default)True | False]
    :return:
    """
    from sklearn.metrics import confusion_matrix

    n_classes = len(classes)
    mat = confusion_matrix(real, predicted)
    mat = mat.astype(int)

    if normalize:
        mat = np.round(mat / mat.sum(axis=1), 2)

    ax = ax or plt.gca()
    plt.rc('font', size=12)  # controls default text sizes
    plt.rc('axes', titlesize=16)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
    plt.rc('legend', fontsize=12)  # legend fontsize
    
    ax.imshow(mat, cmap='seismic')
    for (j, i), label in np.ndenumerate(mat):
        ax.text(i, j, label, ha='center', va='center', color='w')

    #ax.set_title('Confusion matrix on test data')
    ax.set_xticks(np.arange(0, n_classes, 1))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(np.arange(0, n_classes, 1))
    ax.set_yticklabels(classes)
    ax.set_xlabel('predicted')
    ax.set_ylabel('real')

    if doplot:
        plt.show()
    if fig_save_name is not None:
        plt.savefig(fig_save_name, facecolor=fig.get_facecolor(), transparent=True)

    return mat
