# -*- coding: utf-8 -*-
import os
from datetime import datetime
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid


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


def load_image_xarray(fname, K2degC=273.15):

    if isinstance(fname, str):

        dataset = xr.open_dataset(fname)

    elif isinstance(fname, list):

        dataset = xr.open_mfdataset(fname, combine='by_coords')

    sst = dataset['analysed_sst'].sel(
        lat=slice(*latLim), lon=slice(*lonLim)
        ) - K2degC

    sst.attrs['units'] = 'degC'
    sst.attrs['long_name'] = dataset.analysed_sst.attrs['long_name']

    return sst


def get():
    import xarray as xr
    mur = xr.open_dataset(fname, mask_and_scale=True)
    # mur.values, mur.var,  mur.dims, mur.coords, mur.attrs
    sst = mur.analysed_sst.sel(
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


def make_cartopy(
        projection=ccrs.Mercator(),
        resolution='10m', label_step=2,
        **kwargs):

    kw_fig = dict(figsize=(8, 6), facecolor='w')

    kw_fig = {
        key: kwargs.get(key, value)
        for key, value in kw_fig.items()}

    fig, ax = plt.subplots(
        subplot_kw=dict(projection=projection), **kw_fig)

    extent = [*lonLim, *latLim]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.coastlines(resolution=resolution, color='grey', zorder=3)
    # coastline = cfeature.GSHHSFeature(scale='intermediate')
    # ax.add_feature(coastline)

    ax.add_feature(
        cfeature.LAND.with_scale(resolution), facecolor='.85', zorder=2)
    # ax.add_feature(
    #     cfeature.LAKES.with_scale(resolution), edgecolor='grey')
    # ax.stock_img()  # add an underlay image

    draw_labels = (projection == ccrs.PlateCarree() or
                   projection == ccrs.Mercator())

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=draw_labels,
        xlocs=range(
            *map(int, [lonLim[0], lonLim[1] + label_step]),
            label_step),
        ylocs=range(
            *map(int, [latLim[0], latLim[1] + label_step]),
            label_step),
        # xformatter=LONGITUDE_FORMATTER,
        # yformatter=LATITUDE_FORMATTER,
        linestyle='--', linewidth=.8, color='k', alpha=0.2, zorder=3)

    gl.top_labels = gl.right_labels = False

    return fig, ax


def plot_data(da, title=None, ctitle=None,
              lSST=(None, None), SAVE=False, **kwargs):

    # kwargs = dict(figsize=(10, 8), facecolor='w')
    # fig, ax, m = make_map(**kwargs)

    fig, ax = make_cartopy()

    fig.subplots_adjust(left=.03)

    cmap = plt.cm.Spectral_r
    # cmap = cmb.GMT_no_green_r

    cbar_ax = fig.add_axes([.87, .3, .01, .4])

    if da.name is 'analysed_sst':

        da.plot(
            ax=ax, transform=ccrs.PlateCarree(), robust=True,
            vmin=lSST[0], vmax=lSST[1], cmap=cmap,
            cbar_ax=cbar_ax, zorder=1,
            cbar_kwargs={
                'shrink': .4,
                'label': f'{da.name} [{da.units}]',
                'extend': 'both'})

    elif da.name is 'gradient_mag_sst':

        da.plot(
            ax=ax, transform=ccrs.PlateCarree(),
            robust=True, cmap=cmap,
            cbar_ax=cbar_ax, zorder=1,
            cbar_kwargs={
                'shrink': .4,
                'label': f'{da.name} [{da.units}]',
                'extend': 'both'})

    kw_title = dict(fontsize=14, fontweight='roman', pad=10)

    kw_title = {
        key: kwargs.get(key, value)
        for key, value in kw_title.items()}

    title = f'{title}\n{da.time.dt.strftime(r"%d-%m-%Y").values.squeeze()}'

    ax.set_title(title, **kw_title)

    # Batimetria.

    try:

        bathy

    except NameError:

        extract_etopo_bathy()

    levels = [-2000, -1000, -200]

    cs = ax.contour(lonBathy, latBathy, bathy,
                    colors='0.', alpha=.5, levels=levels,
                    linestyles='solid', linewidths=1,
                    transform=ccrs.PlateCarree(), zorder=2)

    ax.clabel(cs, fmt='%1.0f m', inline=1,
              inline_spacing=8, manual=False)

    lonCSM, latCSM = -48.9, -28.6

    if (lonLim[0] < lonCSM < lonLim[1] and
            latLim[0] < latCSM < latLim[1]):

        lonCSMtxt, latCSMtxt = -49.15, -28.6

        ax.plot(lonCSM, latCSM, transform=ccrs.PlateCarree(),
                marker='v', markersize=5, markerfacecolor='r',
                markeredgecolor='.1', markeredgewidth=.7,
                linestyle='None', zorder=5)

        ax.text(lonCSMtxt, latCSMtxt, 'CSM', transform=ccrs.PlateCarree(),
                color='.3', fontsize=13, ha='right', va='center')

    lonCF, latCF = -42.05, -22.88

    if (lonLim[0] < lonCF < lonLim[1] and
            latLim[0] < latCF < latLim[1]):

        lonCFtxt, latCFtxt = -42.2, -22.82

        ax.plot(lonCF, latCF, transform=ccrs.PlateCarree(),
                marker='v', markersize=5, markerfacecolor='r',
                markeredgecolor='.1', markeredgewidth=.7,
                linestyle='None', zorder=5)

        ax.text(lonCFtxt, latCFtxt, 'CF', transform=ccrs.PlateCarree(),
                color='.3', fontsize=13, ha='right', va='bottom')

    if SAVE:

        figname = f'figures/mursst_{date.strftime(r"%Y%m%d")}.png'
        fig.savefig(figname)
        plt.close()

    return fig, ax


def _spheric_gradient_mag(arr, lon, lat, deg2km=111.12):
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


def spheric_gradient_mag(arr, dim=['lon', 'lat'], deg2km=111.12):
    """
    Computes the magnitude of the horizontal gradient vector
    in geographic-like spherical coordinates.
    """
    # deg2km = 111.12

    # Mean latitude of the SST grid in radians.
    mlat = np.mean(lat * np.pi / 180.)

    print(mlat)

    # # Mean zonal spacing of grid.
    # dx = deg2km * np.cos(mlat) * np.mean(np.diff(lon,  axis=0))

    # # Exact meridional spacing of grid.
    # dy = deg2km * np.mean(np.diff(lat, axis=0))

    # # Horizontal gradient.
    # gx, gy = np.gradient(arr, dx, dy)

    # g2 = gx**2 + gy**2

    # return g2


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

    # lSST = temperature_limits(date)

    # for fname in fnames:

    #     # date, lon, lat, sst = load_image_netCDF(fname)
    #     sst = load_image_xarray(fname)

    #     fig, ax = plot_data(
    #         sst,
    #         title='Temperatura da Superfície do Mar')

    #     sstgrad = spheric_gradient_mag(
    #         sst.values.squeeze(), sst.lon.values, sst.lat.values)

    #     sstgrad = xr.DataArray(
    #         sstgrad[np.newaxis, ...],
    #         name='gradient_mag_sst',
    #         dims=['time', 'lat', 'lon'],
    #         coords={'time': sst.time,
    #                 'lat': sst.lat,
    #                 'lon': sst.lon},
    #         attrs={'units': r'degC$^2$ km$^{-2}$',
    #                'long_name': 'magnitude of horizontal gradient'})

    #     fig1, ax1 = plot_data(
    #         sstgrad,
    #         title='Gradiente da Temperatura da Superfície do Mar')

    ds = load_image_xarray(fnames)

    lSST = (
        ds.min(['lat', 'lon']).values.mean(),
        ds.max(['lat', 'lon']).values.mean(),
    )
