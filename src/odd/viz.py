import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories(trajs, title=None, path=None):
    for i, tr in enumerate(trajs):
        tr = np.asarray(tr)
        plt.plot(tr[:,1], tr[:,0], alpha=0.7, label=str(i) if i<8 else None)
    if title: plt.title(title)
    plt.xlabel("lon (norm)"); plt.ylabel("lat (norm)")
    if path:
        plt.savefig(path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
