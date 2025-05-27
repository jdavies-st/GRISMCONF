# Note: Code provided by R. Ryan.

from jwst import datamodels
import asdf
import os, sys
import requests
import numpy as np 
from astropy.modeling import polynomial
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from stpipe.crds_client import get_reference_file


def reformat_poly(obj):
    '''
    Function to transform an astropy.Polynomial object into an list() that matches a given row of a grismconf file
    '''
    
    coefs = list(np.zeros(len(obj.parameters), dtype=float))
    n = len(coefs)

    if isinstance(obj, polynomial.Polynomial1D):
        for i in range(n):
            coefs[i] = [getattr(obj, f'c{i}').value]
    elif isinstance(obj, polynomial.Polynomial2D):
        m = int(np.sqrt(8 * n + 1) - 1) // 2
        i = 0
        for j in range(m):
            for k in range(j + 1):
                coefs[i] = getattr(obj, f'c{j - k}_{k}').value
                i += 1
    else:
        raise NotImplementedError(type(obj))
        
    return coefs

def get_sensitivity(wfss_file, order=1, show=False):
    """Fetch and process the sensitivity file for this observation. This function cleans up the content of 
    the calibration file and changes the units of the sensitivity to be in flam per DN/s"""
    
    with datamodels.open(wfss_file) as dm:
        # Use CRDS to get the photom reference file
        parameters = dm.get_crds_parameters()
        photom = get_reference_file(parameters, "photom", "jwst")
        pupil = dm.meta.instrument.pupil
        filter = dm.meta.instrument.filter

        tab = Table.read(photom)
        pixel_area = tab.meta['PIXAR_SR']
        ok = (tab['filter'] == filter) & (tab['pupil'] == pupil) & (tab['order'] == order)
        w = np.asarray(tab[ok][0]['wavelength'])
        s = np.asarray(tab[ok][0]['relresponse'])
        photmjsr = tab[ok][0]['photmjsr']
        ok = np.nonzero(w)
        w = w[ok]
        s = s[ok]

    # The sensitivity is by default in units of Mjy per SR per DN/s (per pixel) which we convert to
    # the more traditional value of erg/s/cm^2/A per DN/s
    c = 29_979_245_800.0 
    s2 = (w * 1e4) / c * (w / 1e8) / (s * photmjsr * 1e6 * 1e-23 * pixel_area) * 10000

    if show:
        plt.plot(w, s2)
        plt.xlabel(r"Wavelength ($\mu m$)")
        plt.ylabel(r"DN/s per erg/s/cm^2/$\AA$")
        plt.grid()

    return w, s2

def specwcs_poly(wfss_file, order=1):
    DISPX_data = {}
    DISPY_data = {}
    DISPL_data = {}
    SENS_data = {}

    with datamodels.open(wfss_file) as dm:
        t = dm.meta.wcs.get_transform('detector', 'grism_detector')[-1]
        for order, xmodel, ymodel, lmodel in zip(t.orders, t.xmodels, t.ymodels, t.lmodels):
            sorder = f'{order:+}'

            DISPX_data[sorder] = np.array([reformat_poly(p2d) for p2d in xmodel])
            if len(xmodel) == 1:
                DISPX_data[sorder] = DISPX_data[sorder][0]

            DISPY_data[sorder] = np.array([reformat_poly(p2d) for p2d in ymodel])
            if len(ymodel) == 1:
                DISPY_data[sorder] = DISPY_data[sorder][0]

            # The lmodels are (5,) for the 5 orders, not (5, 3) for NIRISS
            try:
                DISPL_data[sorder] = np.array([reformat_poly(p2d) for p2d in lmodel])
            except TypeError:
                DISPL_data[sorder] = np.array(reformat_poly(lmodel))[0]

            SENS_data[sorder] = get_sensitivity(wfss_file, order=order)


    return DISPX_data, DISPY_data, DISPL_data, SENS_data
