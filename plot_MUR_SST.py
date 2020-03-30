# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime
import netCDF4 as nc
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plot


def select_files(date, fpath='MURdata/', pattern='.nc'):

    from glob import glob

    def fname2datetime(sdate):
        return datetime.strptime(
            sdate.split('/')[-1].split('-')[0],
            r'%Y%m%d%H%M%S').replace(hour=0)

    def date2datetime(sdate):
        return datetime.strptime(sdate, r'%d/%m/%Y')

    fnames = glob(f'{fpath}*.nc')

    selected = [fname
                for fname in fnames
                if fname2datetime(fname) >= date2datetime(date[0]) and
                fname2datetime(fname) <= date2datetime(date[-1])]

    return selected


def subset(lon, lat):

    iwhere = np.where(
        (lon >= lonLim[0]) &
        (lon <= lonLim[1]) &
        (lat >= latLim[0]) &
        (lat <= latLim[1]))

    lon = lon[
        iwhere[0][0]:iwhere[0][-1] + 1,
        iwhere[1][0]:iwhere[1][-1] + 1]

    lat = lat[
        iwhere[0][0]:iwhere[0][-1] + 1,
        iwhere[1][0]:iwhere[1][-1] + 1]

    return lon, lat, iwhere


def linear_scale(data, scale_factor, offset):
    return (scale_factor * data + offset)


def load_image(fname, kelvin2celsius=273.15):
    """
    Load SST of MUR dataset from a netCDF file.
    """
    data = nc.Dataset(fname)

    lon, lat = data['lon'][:], data['lat'][:]
    lon, lat = np.meshgrid(lon, lat)
    lon, lat, idx = subset(lon, lat)

    sst = np.squeeze(data['analysed_sst'][:])[
        idx[0][0]:idx[0][-1] + 1,
        idx[1][0]:idx[1][-1] + 1]

    # sst = np.squeeze(data['analysed_sst'][:])
    # kelvin2celsius = 273.15
    # sst -= kelvin2celsius
    return lon, lat, sst


def get():
    import xarray as xr
    mur = xr.open_dataset(fname, mur.mask_and_scale=True)
    # mur.values, mur.var,  mur.dims, mur.coords, mur.attrs
    sst = mur.analysed_sst
    sst = sst.sel(
        lat=slice(*latLim), lon=slice(*lonLim)).dropna(dim='time') - 273.15
    return None


def get_temperature_limits(days, path='data/all_serie/'):
    """
    From a sequence of images gets the maximum and minimum
    value of temperature to control *vmax* and *vmin* in a 
    pcolormesh plot.
    """
    from glob import glob
    Tmax, Tmin = [], []
    for i in days:
        arqs = path + i + '*'
        images = sorted(glob(arqs))
        for image in images:
            nc = Dataset(image)
            lon, lat = nc.variables['lon'][:], \
                       nc.variables['lat'][:]
            lon, lat = np.meshgrid(lon, lat)
            sst = nc.variables['analysed_sst'][:]
            sst = np.squeeze(sst) - 273.15
            lon, lat, sst = subset(lon, lat, sst, 
                                   lonMin, lonMax, 
                                   latMin, latMax)
            Tmax.append(sst.max())
            Tmin.append(sst.min())
    lT = np.round([np.mean(Tmin), np.mean(Tmax)],
                  decimals=1)
    return lT


def make_map(ax):

    m = Basemap(llcrnrlon=lonLim[0],
                llcrnrlat=latLim[0],
                urcrnrlon=lonLim[1],
                urcrnrlat=latLim[1],
                projection='merc',
                resolution='i',
                area_thresh=1000)

    m.ax = ax
    m.drawcoastlines()
    m.fillcontinents(color='.7')

    xwidth, ywidth = m(lonLim[1] - lonLim[0], latLim[1] - latLim[0])

    meridians = np.arange(lonLim[0], lonLim[1] + 1, 2)
    m.drawmeridians(meridians, linewidth=0.2,
                    labels=[0, 0, 0, 1], yoffset=0.005*ywidth)

    parallels = np.arange(latLim[0], latLim[1] + 1, 2)
    m.drawparallels(parallels, linewidth=0.2,
                    labels=[1, 0, 0, 0])

    return m


def plot_sst(fname, lon, lat, sst, lT, show=False):
    if show:
        plt.ion()
    else:
        plt.ioff()

    date_str = fname.split('/')[-1].split('-')[0]
    date_str = date_str[0:4] + '/' + date_str[4:6] + '/' + date_str[6:]

    fig1, ax1 = plt.subplots(figsize=(8,5.7), facecolor='w')
    fig1.subplots_adjust(**kw_adjust)

    m = make_map(ax1)

    mlon, mlat = m(lon, lat)
    mlonBathy, mlatBathy = m(lonBathy, latBathy)

    cmap = plt.cm.Spectral_r
    # cmap = cmb.GMT_no_green_r
    cm = m.pcolormesh(mlon, mlat, sst, cmap=cmap,
                      vmin=lT[0], vmax=lT[1], zorder=1)
    kw_colorbar1 = dict(location='right', size='3%', pad='4%', 
                        extend='both')
    cb = m.colorbar(cm, ax=ax1, **kw_colorbar1)
    cb.ax.text(0.6, 1.07, ur'[°C]', **kw_units)

    title_str = 'Sea Surface Temperature'
    ax1.set_title(title_str, size='x-large', weight='roman')

    levels = [-2000, -1000, -200, -140]
    cs = m.contour(mlonBathy, mlatBathy, bathy, 
                   colors='0.3', levels=levels, 
                   linestyles='solid', zorder=2)

    ax1.clabel(cs, fmt='%1.0f m', inline=1,
               inline_spacing=4, manual=False)

    lonCSM, latCSM = -48.85, -28.6
    lonCSMtxt, latCSMtxt = -49.3, -28.65
    mlonCSM, mlatCSM = m(lonCSM, latCSM)
    mlonCSMtxt, mlatCSMtxt = m(lonCSMtxt, latCSMtxt)

    m.plot(mlonCSM, mlatCSM, marker='*', markersize=10,
           markerfacecolor=[1,0,0], markeredgecolor=[0, 0, 0],
           markeredgewidth=1, linestyle='None', zorder=5)

    m.ax.text(mlonCSMtxt, mlatCSMtxt, 'CSM', color='k', 
              fontsize=14)

    xt, yt = m(-50.8, -27.3)
    ax1.text(xt, yt, date_str,  zorder=6, **kw_text)

    if False: # Chosen line.
        lonStart, latStart = m(-48.0610, -29.8565)
        lonEnd, latEnd = m(-46.8444, -28.4787)
        ax1.plot([lonStart, lonEnd], [latStart, latEnd], 
                  'ok', linestyle='solid')

    if False:
        figname = fname.split('/')[-1][:8] + '.png'
        figname1 = 'figures/events/sst_new%s' % (figname)
        fig1.savefig(figname1)
        plt.close()
    return fig1, ax1, m


if __name__ == "__main__":

    with open('/'.join([os.getcwd(), 'inputDownload.json'])) as finput:

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

    # lon, lat, sst = load_image(fnames[0], lon, lat)
