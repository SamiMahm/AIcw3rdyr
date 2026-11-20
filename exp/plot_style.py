import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

COL_W = 3.35  
PAGE_W = 6.9   


BLUE = "#0c5ec3"
ORANGE = "#c44310"
AQUA = "#0cd08b"
YELLOW = "#fbb92b"
MAGENTA = "#d41c63"
GREY = "#4c4c4a"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
)

DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#5598e7", "#f0efec", "#e66767", "#a32121"]
)


def apply():

    mpl.rcParams.update({
        "font.family": "serif",      
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GREY,
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": GREY,
        "ytick.color": GREY,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,           
    })

#voor de save
def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.name}")
