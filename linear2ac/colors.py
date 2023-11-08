import seaborn as sns
import pandas as pd

def cueset_color(cuesets,alpha=1):
    """Return list of colors based on cueset name
    """
    color_list = {
        'Cue Set A': list(sns.color_palette("colorblind")[0])+[alpha],
        'Cue Set B': list(sns.color_palette("colorblind")[1])+[alpha],
        'Cue Set C': list(sns.color_palette("colorblind")[2])+[alpha],
        'Cue Set D': list(sns.color_palette("colorblind")[3])+[alpha],
        'Cue Set E': list(sns.color_palette("colorblind")[4])+[alpha],
    }
    if (type(cuesets) is list) | (type(cuesets) is pd.Series) :
            return [color_list[cueset] for cueset in cuesets]
    else:
        return color_list[cuesets]