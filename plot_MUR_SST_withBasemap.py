# -*- coding: utf-8 -*-
import os
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
        np.logical_and(lon >= lonLim[0], lon <= lonLim[1])
    )

    ilat, = np.where(
        np.logical_and(lat >= latLim[0], lat <= latLim[1])
    )

    return ilon, ilat


def linear_scale(data, scale_factor, offset):
    return (scale_factor * data + offset)


def load_image_netCDF(fname, K2degC=273.15, scale=False):
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
    # import xarray as xr

    # mur = xr.open_dataset(fname, mur.mask_and_scale=True)
    # mur.values, mur.var,  mur.dims, mur.coords, mur.attrs
    # mur = dataset.analysed_sst

    dataset = xr.open_dataset(fname)

    sst = dataset['analysed_sst'].sel(
        lat=slice(*latLim), lon=slice(*lonLim)
        ).drop('time') - K2degC

    return lon, lat, sst


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


def get_etopo_indexes(lon, lat, order='C'):

    if order == 'C':

        lonStart = np.squeeze(np.where(lon >= lonLim[0]))[0]
        lonEnd = np.squeeze(np.where(lon > lonLim[1]))[1]

        lat = np.flip(lat)
        latStart = np.squeeze(np.where(lat <= latLim[1]))[0]
        latEnd = np.squeeze(np.where(lat < latLim[0]))[1]

    if order == 'F':

        lonStart = np.squeeze(np.where(lon > lonLim[0]))[0]
        lonEnd = np.squeeze(np.where(lon > lonLim[1]))[1]

        latStart = np.squeeze(np.where(lat < latLim[0]))[-1]
        latEnd = np.squeeze(np.where(lat > latLim[1]))[0]

    return lonStart, lonEnd, latStart, latEnd


def extract_etopo_bathy(
        fname='../etopo1/ETOPO1_Ice_g_gdal.grd.gz',
        order='C'):
    # https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/'
    # data/ice_surface/grid_registered/netcdf/
    # ETOPO1_Ice_g_gdal.grd.gz
    import gzip

    global lonBathy, latBathy, bathy

    with gzip.open(fname) as gzfile:
        with nc.Dataset(
            'justafakename', mode='r', memory=gzfile.read()
        ) as ncfile:
            dxdy = ncfile['spacing'][0]
            lon = np.arange(*ncfile['x_range'][:] + [0, dxdy], dxdy)
            lat = np.arange(*ncfile['y_range'][:] + [0, dxdy], dxdy)
            lonStart, lonEnd, latStart, latEnd = get_etopo_indexes(
                lon, lat, order=order)

            z = ncfile['z']

            if order == 'C':
                zz = np.reshape(z, ncfile['dimension'][::-1], order='C')

            elif order == 'F':
                zz = np.transpose(
                    np.reshape(z, ncfile['dimension'][:], order='F'))

    if order == 'C':

        # lat = np.flip(lat)

        lonBathy, latBathy = np.meshgrid(
            lon[lonStart:lonEnd],
            np.flip(lat)[latStart:latEnd]
        )

        bathy = zz[latStart:latEnd, lonStart:lonEnd]

    if order == 'F':

        lonBathy, latBathy = np.meshgrid(
            lon[lonStart:lonEnd],
            lat[latStart:latEnd]
        )

        bathy = np.transpose(
            np.flip(zz, 1)[lonStart:lonEnd, latStart:latEnd]
        )

    return None


def get_cube(fname='../etopo1/ETOPO1_Ice_g_gdal.grd.gz'):
    import iris
    coord_values = {'latitude': lambda cell: latLim[0] <= cell <= latLim[1],
                    'longitude': lambda cell: lonLim[0] <= cell <= lonLim[1]}
    constraint = iris.Constraint(coord_values=coord_values)
    bathy = iris.load_cube(fname, constraint)
    return bathy


def make_basemap(projection='merc', resolution='c', **kwargs):

    kw_fig = dict(figsize=(8, 6), facecolor='w')

    kw_fig = {
        key: kwargs.get(key, value)
        for key, value in kw_fig.items()}

    fig, ax = plt.subplots(**kw_fig)

    m = Basemap(llcrnrlon=lonLim[0],
                llcrnrlat=latLim[0] - 1e-7,
                urcrnrlon=lonLim[1],
                urcrnrlat=latLim[1] + 1e-7,
                projection=projection,
                resolution=resolution,
                area_thresh=10)

    m.ax = ax
    m.drawcoastlines()
    m.fillcontinents(color='.7', lake_color='blue')
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


def plot_data(
        date, lon, lat, sst,
        title=None, ctitle=None,
        lSST=(None, None), **kwargs):

    # kwargs = dict(figsize=(10, 8), facecolor='w')
    # fig, ax, m = make_map(**kwargs)

    fig, ax, m = make_basemap(resolution='h')

    fig.subplots_adjust(left=.03)

    x, y = m(*np.meshgrid(lon, lat))

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

    # Batimetria.

    try:

        xBathy, yBathy = m(lonBathy, latBathy)

    except NameError:

        extract_etopo_bathy()

        xBathy, yBathy = m(lonBathy, latBathy)

    levels = [-2000, -1000, -200]

    cs = m.contour(xBathy, yBathy, bathy,
                   colors='0.', levels=levels,
                   linestyles='solid', linewidths=.8,
                   zorder=2)

    ax.clabel(cs, fmt='%1.0f m', inline=1,
              inline_spacing=4, manual=False)

    lonCSM, latCSM = -48.9, -28.6

    if (lonLim[0] < lonCSM < lonLim[1] and
            latLim[0] < latCSM < latLim[1]):

        lonCSMtxt, latCSMtxt = -49.15, -28.6
        xCSM, yCSM = m(lonCSM, latCSM)
        xCSMtxt, yCSMtxt = m(lonCSMtxt, latCSMtxt)

        m.plot(xCSM, yCSM,
               marker='v', markersize=5,
               markerfacecolor='r', markeredgecolor='w',
               markeredgewidth=1.2, linestyle='None', zorder=5)

        m.ax.text(xCSMtxt, yCSMtxt, 'CSM', color='w',
                  fontsize=13, ha='right', va='center')

    lonCF, latCF = -42.05, -22.88

    if (lonLim[0] < lonCF < lonLim[1] and
            latLim[0] < latCF < latLim[1]):

        lonCFtxt, latCFtxt = -42.2, -22.82
        xCF, yCF = m(lonCF, latCF)
        xCFtxt, yCFtxt = m(lonCFtxt, latCFtxt)

        m.plot(xCF, yCF,
               marker='v', markersize=5,
               markerfacecolor='r', markeredgecolor='w',
               markeredgewidth=1.2, linestyle='None', zorder=5)

        m.ax.text(xCFtxt, yCFtxt, 'CF', color='w',
                  fontsize=13, ha='right', va='bottom')

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

    # Mean latitude of the SST grid in radians.
    mlat = np.mean(lat * np.pi / 180.)

    # Mean zonal spacing of grid.
    dx = deg2km * np.cos(mlat) * np.mean(np.diff(lon,  axis=0))

    # Exact meridional spacing of grid.
    dy = deg2km * np.mean(np.diff(lat, axis=0))

    # Horizontal gradient.
    gx, gy = np.gradient(arr, dx, dy)

    g2 = gx**2 + gy**2

    return g2


if __name__ == "__main__":

    import json

    with open(
            '/'.join([os.getcwd(), 'inputDownload.json'])) as finput:

        parsed_json = json.load(finput)

        date = parsed_json['date']
        lonLim = parsed_json['lon']
        latLim = parsed_json['lat']

    lonLim = [float(l) for l in lonLim]
    latLim = [float(l) for l in latLim]

    # Sorting for use.
    lonLim.sort()
    latLim.sort()

    fnames = select_files(date)

    lSST = temperature_limits(date)

    for fname in fnames:

        date, lon, lat, sst = load_image_netCDF(fname)
        # date, lon, lat, sst = load_image_xarray(fname)

        fig, ax, m = plot_data(
            date, lon, lat, sst,
            title='Temperatura da Superfície do Mar',
            ctitle=r'[°C]')

        sstgrad = spheric_gradient_mag(sst, lon, lat)

        fig1, ax1, m1 = plot_data(
            date, lon, lat, sstgrad,
            title='Gradiente da Temperatura da Superfície do Mar',
            ctitle=r'[°C$^2$ km$^{-2}$]')
