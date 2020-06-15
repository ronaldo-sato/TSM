# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime
import netCDF4 as nc
import xarray as xr
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def select_files(date=None, fpath='MURdata/', pattern='.nc'):

    from glob import glob

    def fname2datetime(sdate):
        return datetime.strptime(
            sdate.split('/')[-1].split('-')[0],
            r'%Y%m%d%H%M%S').replace(hour=0)

    def date2datetime(sdate):
        return datetime.strptime(sdate, r'%d/%m/%Y')

    fnames = glob(f'{fpath}*.nc')

    if date:

        fnames = [
            fname
            for fname in fnames
            if fname2datetime(fname) >= date2datetime(date[0]) and
            fname2datetime(fname) <= date2datetime(date[-1])]

    return sorted(fnames)


def subset_indexes(lon, lat):

    if not isinstance(lon, np.ndarray):
        lon = lon[:]

    if not isinstance(lat, np.ndarray):
        lat = lat[:]

    ilon, = np.where(
        (lon >= lonLim[0]) &
        (lon <= lonLim[1])
    )

    ilat, = np.where(
        (lat >= latLim[0]) &
        (lat <= latLim[1])
    )

    return ilon, ilat


def linear_scale(data, scale_factor, offset):
    return (scale_factor * data + offset)


def load_image(fname, K2degC=273.15, scale=False):
    """
    Load MUR dataset Sea Surface Temperatura from a netCDF file.
    """
    ncfile = nc.Dataset(fname)

    # ncfile.variables   # all variable objects
    # ncfile.dimensions  # all dimension objects

    # sst = ncfile['analysed_sst']  # variable object for sst
    # sst.ncattrs()                 # list all variable attributes

    date = nc.num2date(
        ncfile['time'][:], ncfile['time'].units
    )[0]

    lon, lat = ncfile['lon'], ncfile['lat']
    ilon, ilat = subset_indexes(lon, lat)

    if scale:

        sst = ncfile['analysed_sst']

        _sst = linear_scale(
            np.squeeze(sst[:, ilat, ilon]),
            sst.scale_factor,
            sst.add_offset
        )

    else:

        sst = np.squeeze(ncfile['analysed_sst'][:, ilat, ilon])

    return date, lon[ilon], lat[ilat], sst - K2degC


def load_image_xrarray(fname, K2degC=273.15):

    dataset = xr.open_dataset(fname)

    sst = dataset['analysed_sst'].sel(
        lat=slice(*latLim), lon=slice(*lonLim)).drop('time') - K2degC

    return lon, lat, sst


def get():
    import xarray as xr
    # mur = xr.open_dataset(fname, mur.mask_and_scale=True)
    # mur.values, mur.var,  mur.dims, mur.coords, mur.attrs
    sst = mur.analysed_sst
    sst = sst.sel(
        lat=slice(*latLim), lon=slice(*lonLim)).dropna(dim='time') - 273.15
    return None


def temperature_limits(days=None, fpath='MURdata/'):
    """
    From a sequence of images gets the maximum and minimum
    temperature values in degrees Celsius to control *vmax*
    and *vmin* in pcolormesh plot.
    """
    fnames = select_files(days, fpath=fpath)

    sstMax, sstMin = [], []

    for fname in fnames:

        ncfile = nc.Dataset(fname)

        sst = ncfile.variables['analysed_sst']

        if fname is fnames[0]:

            ilon, ilat = subset_indexes(ncfile['lon'], ncfile['lat'])

        sst = np.squeeze(sst[:, ilat, ilon]) - 273.15

        sstMax.append(sst.max())
        sstMin.append(sst.min())

    lT = np.round(
        [np.mean(sstMin), np.mean(sstMax)],
        decimals=1)

    return lT


def extract_topo_isolines(fname='etopo1.tiff'):
    tiff = plt.imread(fname)
    plt.imshow(tiff)
    return x, y, z


def make_map(**kwargs):

    kw_fig = dict(figsize=(8, 6), facecolor='w')

    kw_fig = {
        key: kwargs.get(key, value)
        for key, value in kw_fig.items()}

    fig, ax = plt.subplots(**kw_fig)

    m = Basemap(llcrnrlon=lonLim[0],
                llcrnrlat=latLim[0] - 1e-4,
                urcrnrlon=lonLim[1],
                urcrnrlat=latLim[1] + 1e-4,
                projection='merc',
                resolution='h')

    m.ax = ax
    m.drawcoastlines()
    m.fillcontinents(color='.7', lake_color='azure')
    # m.shadedrelief(scale=.5)
    # m.etopo(scale=.5)

    xwidth, ywidth = m(lonLim[1] - lonLim[0], latLim[1] - latLim[0])

    meridians = np.arange(lonLim[0], lonLim[1] + 1, 2)

    xticks = m.drawmeridians(meridians, linewidth=0.3,
                             labels=[0, 0, 0, 1], yoffset=.004*ywidth)

    parallels = np.arange(latLim[0], latLim[1] + 1, 2)

    yticks = m.drawparallels(parallels, linewidth=0.3,
                             labels=[1, 0, 0, 0])

    return fig, ax, m


def plot_map(
        date, lon, lat, sst, 
        title=None, ctitle=None,
        lSST=(None, None), **kwargs):

    # kwargs = dict(figsize=(10, 8), facecolor='w')
    # fig, ax, m = make_map(**kwargs)

    fig, ax, m = make_map()

    fig.subplots_adjust(left=.03)

    x, y = m(*np.meshgrid(lon, lat))
    # mlonBathy, mlatBathy = m(lonBathy, latBathy)

    cmap = plt.cm.Spectral_r
    # cmap = cmb.GMT_no_green_r

    cm = m.pcolormesh(x, y, sst, cmap=cmap,
                      vmin=lSST[0], vmax=lSST[1], zorder=1)

    kw_colorbar = dict(location='right', size='100%', pad='0%',
                       extend='both')
    # kw_colorbar = dict(orientation='vertical',
    #                    fraction=.018, pad=.03, shrink=1.2, extend='both')

    kw_colorbar = {
        key: kwargs.get(key, value)
        for key, value in kw_colorbar.items()}

    cax = fig.add_axes([.85, .3, .02, .4])
    _ = cax.axis('off')

    cbar = m.colorbar(cm, ax=cax, **kw_colorbar)
    # cbar = plt.colorbar(cm, ax=ax, **kw_colorbar)

    kw_units = dict(fontsize=12, fontweight='roman',
                    horizontalalignment='center',
                    verticalalignment='top')

    kw_units = {
        key: kwargs.get(key, value)
        for key, value in kw_units.items()}

    # cbar.set_label(ctitle, **kw_units)

    cbar.ax.set_title(
        ctitle, horizontalalignment='left', x=-.2, y=1.01)

    kw_title = dict(fontsize=14, fontweight='roman', pad=10)

    kw_title = {
        key: kwargs.get(key, value)
        for key, value in kw_title.items()}

    title = f'{title}\n{date.strftime(r"%d-%m-%Y")}'

    ax.set_title(title, **kw_title)
    
    # fig.suptitle(f'{date.strftime(r"%d-%m-%Y")}',
    #              y=.9, fontsize=14, fontweight='roman')

    # levels = [-2000, -1000, -200, -140]
    # cs = m.contour(mlonBathy, mlatBathy, bathy,
    #                colors='0.3', levels=levels,
    #                linestyles='solid', zorder=2)

    # ax.clabel(cs, fmt='%1.0f m', inline=1,
    #           inline_spacing=4, manual=False)

    # lonCSM, latCSM = -48.85, -28.6
    # lonCSMtxt, latCSMtxt = -49.3, -28.65
    # mlonCSM, mlatCSM = m(lonCSM, latCSM)
    # mlonCSMtxt, mlatCSMtxt = m(lonCSMtxt, latCSMtxt)

    # m.plot(mlonCSM, mlatCSM, marker='*', markersize=10,
    #        markerfacecolor=[1, 0, 0], markeredgecolor=[0, 0, 0],
    #        markeredgewidth=1, linestyle='None', zorder=5)

    # m.ax.text(mlonCSMtxt, mlatCSMtxt, 'CSM', color='k', 
    #           fontsize=14)

    if False:

        figname = f'figures/mursst_{date.strftime(r"%Y%m%d")}.png'
        fig.savefig(figname)
        plt.close()

    return fig, ax, m


def spheric_gradient_mag(arr, lon, lat, deg2km=111.12):
    """
    Computes the magnitude of the horizontal gradient vector
    in geographic-like spherical coordinates.
    """
    # deg2km = 111.12
    # Mean latitude of the SST grid.
    mlat = np.mean(lat * np.pi / 180.)
    # Mean zonal spacing of grid.
    dx = deg2km * np.cos(mlat) * np.mean(np.diff(lon,  axis=0))
    # Exact meridional spacing of grid.
    dy = deg2km*np.mean(np.diff(lat, axis=0))

    # Horizontal gradient.
    gx, gy = np.gradient(arr, dx, dy)
    g2 = gx**2 + gy**2

    return g2


if __name__ == "__main__":

    with open(
            '/'.join([os.getcwd(), 'inputDownload.json'])) as finput:

        parsed_json = json.load(finput)

        date = parsed_json['date']
        lonLim = parsed_json['lon']
        latLim = parsed_json['lat']

    fnames = select_files(date)

    lonLim = [float(l) for l in lonLim]
    latLim = [float(l) for l in latLim]

    # Sorting for use.
    lonLim.sort()
    latLim.sort()

    lSST = temperatura_limits()

    for fname in fnames:

        date, lon, lat, sst = load_image(fname)

        fig, ax, m = plot_map(
            date, lon, lat, sst,
            title='Temperatura da Superfície do Mar',
            ctitle=r'[°C]')

        sstgrad = spheric_gradient_mag(sst, lon, lat)

        fig1, ax1, m1 = plot_map(
            date, lon, lat, sstgrad,
            title='Gradiente da Temperatura da Superfície do Mar',
            ctitle=r'[°C$^2$ km$^{-2}$]')
