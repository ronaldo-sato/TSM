# -*- coding: utf-8 -*-
from __future__ import division
import json
import os
from datetime import datetime
from datetime import timedelta
import urllib
# import threading


def gen_dates(date, hour=9):
    u"""
    Recebe uma lista de datas (início e final ou apenas um dia)
    e converte a data ou intervalo de datas em uma lista
    de objetos datetime.
    """
    from dateutil import rrule
    if len(date) > 1 and all([len(item) for item in date]):
        start = datetime.strptime(date[0], r'%d/%m/%Y')
        end = datetime.strptime(date[1], r'%d/%m/%Y')
        dtList = list(
            rrule.rrule(freq=rrule.DAILY, dtstart=start, interval=1, until=end)
        )
        return [d.replace(hour=hour) for d in dtList]
    else:
        return [
            datetime.strptime(d, r'%d/%m/%Y').replace(hour=hour)
            for d in filter(None, date)]


def month_last_day(day):
    u"""
    Recebe um objeto datetime (uma data) e retorna o último dia
    do mês da respectiva data.
    """
    from dateutil.relativedelta import relativedelta

    next_month = day.replace(day=1) + relativedelta(months=1)

    return next_month - timedelta(days=next_month.day)


def boundingAreaIndex(lonLim, latLim,
                      lonRange=[-180, 180], lonRes=.01,
                      latRange=[-90, 90], latRes=.01):
    u"""
    Recebe os limites geográficos da área de interesse e
    devolve os índices correspondentes das coordenadas.

    OBS: Argumentos pré-definidos para MUR SST.
    """

    import numpy as np

    lonLim, latLim = [float(l) for l in lonLim], [float(l) for l in latLim]

    lonLim.sort()
    latLim.sort()

    ilon = [int((lonLim[0] - lonRange[0] - lonRes) / lonRes),
            int((lonLim[1] - lonRange[0]) / lonRes - 1)]

    ilat = [int((latLim[0] - latRange[0] - latRes) / latRes),
            int((latLim[1] - latRange[0]) / latRes - 1)]

    return [ilon, ilat]


if __name__ == "__main__":

    # Discover Datasets And More In: https://gcmd.gsfc.nasa.gov/

    ncTail = '-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'

    with open('/'.join([os.getcwd(), 'inputDownload.json'])) as finput:

        parsed_json = json.load(finput)

        date = parsed_json['date']
        lonLim = parsed_json['lon']
        latLim = parsed_json['lat']

        if parsed_json['url']:

            url = parsed_json['url']

        else:

            url = ('https://opendap.jpl.nasa.gov/'
                   'opendap/allData/ghrsst/data/'
                   'GDS2/L4/GLOB/JPL/MUR/v4.1/')

    dtDate = gen_dates(date)

    ilon, ilat = boundingAreaIndex(lonLim, latLim)

    idx = f'[0:1:0][{ilat[0]}:1:{ilat[1]}][{ilon[0]}:1:{ilon[1]}]'

    for dt in dtDate:

        urlTail = '{}/{:03d}/'.format(dt.year, dt.timetuple().tm_yday)

        ncHead = '{}'.format(dt.strftime(r'%Y%m%d%H%M%S'))

        fname = f'{ncHead}{ncTail}'

        furl = f'{url}{urlTail}{fname}?analysed_sst{idx}'

        # Exploring Information About Data Using Browser:
        # Paste the url in a browser or open the link from
        # the terminal (ctrl + click).

        # Dataset Descriptor Structure (dds).
        dds = '{}.dds'.format(furl.split('?')[0])

        # Data Attribute Structure (das).
        das = '{}.das'.format(furl.split('?')[0])

        # Dataset Information (All Information).
        info = '{}.info'.format(furl.split('?')[0])

        # Using urllib to download.
        try:

            urllib.request.urlretrieve(furl, filename=f'MURdata/{fname}')

            # Using the terminal command "wget" to download.

            # cmd = f'wget {furl} -O MURdata/{fname}'

            # os.system(cmd)

        except HTTPError:

            raise(f'Verifique:\n\n\t{url}\n\n\t{ncTail}')
