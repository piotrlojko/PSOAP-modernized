import os
import numpy as np

import psoap
from psoap import constants as C
from psoap.input_parsing import load_spectrum_array


def compute_barycentric_corrections(dates, ra, dec):
    '''
    Compute the barycentric radial-velocity correction for each observation.

    The correction accounts for the motion of the Earth (geocenter) relative
    to the Solar System barycenter, projected along the line of sight to the
    target.  The observatory's exact location on the Earth is *not* taken into
    account (geocentric approximation).

    Args:
        dates (array-like): Barycentric Julian Dates (BJD_TDB) of each
            observation.
        ra (float): Right Ascension of the target in decimal degrees (ICRS).
        dec (float): Declination of the target in decimal degrees (ICRS).

    Returns:
        np.ndarray: Array of velocity corrections in km/s, one per epoch.
            Adding this value to a measured radial velocity converts it from
            the geocentric to the barycentric frame.  The same correction is
            applied as a wavelength *red*-shift when moving observed spectra
            to the barycentric frame.
    '''
    from astropy.coordinates import SkyCoord, EarthLocation
    from astropy.time import Time
    import astropy.units as u

    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    times = Time(np.atleast_1d(dates), format='jd', scale='tdb')
    # Use a geodetic location at zero longitude/latitude with zero elevation so
    # that only the Earth's bulk orbital motion is captured and no observatory-
    # specific correction is applied (per the geocentric approximation).
    location = EarthLocation.from_geodetic(lon=0 * u.deg, lat=0 * u.deg,
                                           height=0 * u.m)
    corrections = sc.radial_velocity_correction(
        kind='barycentric', obstime=times, location=location
    )
    return corrections.to(u.km / u.s).value


def redshift(wl, v):
    '''
    Redshift a vector of wavelengths. A positive velocity corresponds to a
    lengthening (increase) of the wavelengths in the array.

    Args:
        wl (np.array, arbitrary shape): the input wavelengths
        v (float): the velocity by which to redshift the wavelengths [km/s]

    Returns:
        np.array: A redshifted version of the wavelength vector
    '''

    wl_red = wl * np.sqrt((C.c_kms + v) / (C.c_kms - v))
    return wl_red


def lredshift(lwl, v):
    '''
    Redshift a vector of wavelengths already in log-lambda (natural log).
    A positive velocity corresponds to a lengthening (increase) of the
    wavelengths in the array.

    Args:
        lwl (np.array, arbitrary shape): the input ln(wavelengths).
        v (float): the velocity by which to redshift the wavelengths [km/s]

    Returns:
        np.array: A redshifted version of the log-wavelength vector
    '''

    lwl_red = lwl + v / C.c_kms
    return lwl_red


def replicate_wls(lwls, velocities, mask):
    '''
    Using the set of velocities calculated from an orbit, copy and *blue*-shift
    the input ln(wavelengths), so that they correspond to the rest-frame
    wavelengths of the individual components. This routine is primarily for
    producing replicated ln-wavelength vectors ready to feed to the GP
    routines.

    Args:
        lwls (1D np.array with length ``n_good_pixels``): the 1D representation
            of the natural log of the (masked) input wavelength vectors.
        velocities (2D np.array with shape ``(n_components, n_epochs)``): a
            set of velocities determined from an orbital model.
        mask : the np.bool mask used to select the good datapoints. It is
            necessary for properly replicating the velocities to the right
            epoch.

    Returns:
        np.array: A 2D ``(n_components, n_good_pix)`` array of the wavelength
            vectors *blue*-shifted according to the velocities.
    '''

    n_components, n_epochs = velocities.shape

    n_good_pix = np.sum(mask)

    lwls_out = np.empty((n_components, n_good_pix), dtype=np.float64)
    for i in range(n_components):
        lwls_out[i] = lredshift(
            lwls,
            (-velocities[i][:, np.newaxis] * np.ones_like(mask))[mask]
        )

    return lwls_out


class Chunk:
    '''
    Hold a chunk of spectral data. Each chunk is shape ``(n_epochs, n_pix)``
    and has components ``wl``, ``fl``, ``sigma``, ``date``, and ``mask``.

    Input spectra are expected as normalized 1D spectra: wavelength in
    Angstroms, normalized flux (continuum at 1) in arbitrary units, and flux
    error.
    '''

    def __init__(self, wl, fl, sigma, date, mask=None):
        self.wl = wl           #: wavelength vector [Angstroms]
        self.lwl = np.log(wl)  #: natural log of the wavelength vector
        self.fl = fl           #: normalized flux vector
        self.sigma = sigma     #: measurement uncertainty vector
        self.date = date       #: date array (n_epochs, n_pix)
        self.date1D = date[:, 0]  #: date vector of length ``n_epochs``

        if mask is None:
            self.mask = np.ones_like(self.wl, dtype=bool)
        else:
            self.mask = mask
        self.n_epochs, self.n_pix = self.wl.shape

    def apply_barycentric_correction(self, v_bary):
        '''
        Shift each epoch's wavelengths to the barycentric rest frame.

        This is a *preprocessing* step that should be called before
        :meth:`apply_mask`.  After the correction the stored wavelengths
        represent the target's frame shifted only by the orbital motion, not
        by the Earth's annual motion.

        Args:
            v_bary (array-like of float, length ``n_epochs``): Barycentric
                velocity correction in km/s for each epoch (as returned by
                :func:`compute_barycentric_corrections`).  A positive value
                means the Earth is moving towards the target, so the observed
                wavelengths are blue-shifted and must be red-shifted back.
        '''
        v_bary = np.asarray(v_bary)
        for i in range(self.n_epochs):
            self.wl[i] = redshift(self.wl[i], v_bary[i])
            self.lwl[i] = lredshift(self.lwl[i], v_bary[i])

    def apply_mask(self):
        '''
        Apply the mask to all of the attributes, returning 1D arrays.
        '''
        self.wl = self.wl[self.mask]
        self.lwl = self.lwl[self.mask]
        self.fl = self.fl[self.mask]
        self.sigma = self.sigma[self.mask]
        self.date = self.date[self.mask]
        self.N = len(self.wl)

    @classmethod
    def from_textfiles(cls, filenames, dates, limit=None, wl_min=None, wl_max=None):
        '''
        Load spectra from plain-text files with three columns:
        wavelength (Angstroms), normalized flux, and flux error.

        Args:
            filenames (list of str): paths to the text files, one per epoch.
            dates (array-like): observation dates (JD) for each epoch.
            limit (int, optional): maximum number of epochs to load.
            wl_min (float, optional): minimum wavelength to include [Angstroms].
            wl_max (float, optional): maximum wavelength to include [Angstroms].

        Notes:
            When a spectrum's wavelength range does not fully cover ``wl_min``–
            ``wl_max``, interpolation is performed with ``fill_value=1.0`` for flux
            (appropriate for normalized continuum spectra) and ``fill_value=np.inf``
            for sigma (effectively masking those pixels during GP evaluation).

        Returns:
            Chunk: the instantiated Chunk object.
        '''
        from scipy.interpolate import interp1d

        dates = np.atleast_1d(dates)
        if limit is not None:
            filenames = list(filenames)[:limit]
            dates = dates[:limit]

        n_epochs = len(filenames)

        # Load first spectrum to determine pixel grid
        spec0 = load_spectrum_array(filenames[0])
        wl_ref = spec0[:, 0].astype(np.float64)
        mask_cut = np.ones(len(wl_ref), dtype=bool)
        if wl_min is not None:
            mask_cut &= wl_ref >= wl_min
        if wl_max is not None:
            mask_cut &= wl_ref <= wl_max
        wl0 = wl_ref[mask_cut]
        n_pix = len(wl0)

        wl_arr = np.empty((n_epochs, n_pix), dtype=np.float64)
        fl_arr = np.empty((n_epochs, n_pix), dtype=np.float64)
        sigma_arr = np.empty((n_epochs, n_pix), dtype=np.float64)

        for i, fname in enumerate(filenames):
            spec = load_spectrum_array(fname)
            wl_i = spec[:, 0].astype(np.float64)
            fl_i = spec[:, 1].astype(np.float64)
            sigma_i = spec[:, 2].astype(np.float64)

            # Interpolate onto common wavelength grid
            f_fl = interp1d(wl_i, fl_i, bounds_error=False, fill_value=1.0)
            f_sig = interp1d(wl_i, sigma_i, bounds_error=False, fill_value=np.inf)

            wl_arr[i] = wl0
            fl_arr[i] = f_fl(wl0)
            sigma_arr[i] = f_sig(wl0)

        # Broadcast dates to (n_epochs, n_pix)
        date_arr = dates[:, np.newaxis] * np.ones((n_epochs, n_pix))

        print("Loaded {:d} epochs, {:d} pixels each.".format(n_epochs, n_pix))
        return cls(wl_arr, fl_arr, sigma_arr, date_arr)

    def save_textfiles(self, prefix="", suffix=""):
        '''
        Save each epoch as a plain-text three-column file
        (wavelength [Angstroms], normalized flux, sigma).

        Args:
            prefix (str): path prefix for the output files.
            suffix (str): optional suffix appended before the ``.txt`` extension.
        '''
        for i in range(self.n_epochs):
            fname = "{:}epoch_{:03d}{:}.txt".format(prefix, i, suffix)
            np.savetxt(
                fname,
                np.column_stack([self.wl[i], self.fl[i], self.sigma[i]]),
                header="wavelength[AA]  flux  sigma",
                fmt="%.8f  %.8f  %.8f",
            )


# Location of packaged data
basedir = os.path.dirname(psoap.__file__)
