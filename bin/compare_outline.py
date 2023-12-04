import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import read_tiff_orion

color = px.colors.qualitative.Plotly
color[0] = "black"
color[7] = "white"

def binarize(img):
    res = (img - np.min(img)) / (np.max(img) - np.min(img))
    res[res < .5] = 0
    res[res >= .5] = 1
    return res


def compare(outline1, outline2):
    return 100 * np.sum(binarize(outline1) - binarize(outline2)) / np.prod(outline1.shape) 

def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        if j: y += int(j)<<i
    return y


def masks_comparison(list_of_masks, names=[]):
    result = np.apply_along_axis(bool2int, 0, np.stack(list_of_masks).astype(bool))
    if len(list_of_masks) == 3:
        if not names:
            names = range(len(list_of_masks))
        names = {0: 'pas de masque', 1: names[0], 2: names[1], 
                3: f'{names[0]} & {names[1]}', 4: names[2], 
                5: f'{names[0]} & {names[2]}', 6: f'{names[1]} & {names[2]}',
                7: "commun"}
        colorscale = [[0, 'black'], [0.05, 'black'], 
                      [0.05, '#EF553B'], [0.22, '#EF553B'], 
                      [0.22, '#00CC96'], [0.35, '#00CC96'], 
                      [0.35, '#AB63FA'], [0.5, '#AB63FA'], 
                      [0.5, '#FFA15A'], [0.63, '#FFA15A'], 
                      [0.63, '#19D3F3'], [0.77, '#19D3F3'], 
                      [0.77, '#FF6692'], [0.95, '#FF6692'], 
                      [0.95, 'white'], [1, 'white']]
    else:
        colorscale="sunset"
    fig = go.Figure(go.Heatmap(z=result[::-1, :], colorscale=colorscale, colorbar=dict(tickvals=list(names.keys()), ticktext=list(names.values()))))
    fig.show('browser')

if __name__ == "__main__":
    import sys
    print(sys.argv[1:])
    list_masks = [read_tiff_orion(x)[0] for x in sys.argv[1:]]
    masks_comparison(list_masks, names=["sans normalisation", "automatique", "manuelle"])
    # o1 = read_tiff_orion(sys.argv[1])
    # o2 = read_tiff_orion(sys.argv[2])
    # print(compare(o1[-1], o2[-1]))
