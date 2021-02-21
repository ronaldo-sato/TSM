# -*- coding: utf-8 -*-
# from flask import Flask
# import io
# import base64
# import xarray as xr

# app = Falsk(__name__)

# ds = xr.open_mfdataset(
#     'MURdata/*.nc',
#     concat_dim='time',
#     engine='h5netcdf',
#     parallel=True)



# ##
# import io
# import random
# from flask import Response
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure

# @app.route('/plot.png')
# def plot_png():
#     fig = create_figure()
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')

# def create_figure():
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
#     xs = range(100)
#     ys = [random.randint(1, 50) for x in xs]
#     axis.plot(xs, ys)
#     return fig

##
from flask import Flask
#from flask import render_template
import matplotlib.pyplot as plt
import io
import base64

from matplotlib.animation import FuncAnimation
import numpy as np

from IPython.display import HTML

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

app = Flask(__name__)

@app.route('/plot')
def build_plot():

    img = io.BytesIO()

    # y = [1,2,3,4,5]
    # x = [0,2,1,3,4]
    # plt.plot(x,y)
    # plt.savefig(img, format='png')
    # img.seek(0)

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'ro')

    anim = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                         init_func=init, blit=True)
    plt.savefig(fig, format='png')
    # anim.save('anim.mp4')

    img.seek(0)

    # plot_url = base64.b64encode(img.getvalue()).decode()

    plot_url = HTML(anim.to_jshtml())

    return '<img src="data:image/png;base64,{}">'.format(plot_url)

if __name__ == '__main__':

    app.run(debug=True)

