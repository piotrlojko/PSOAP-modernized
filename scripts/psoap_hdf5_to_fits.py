#!/usr/bin/env python

from __future__ import print_function

import argparse
import os

import h5py
import numpy as np
from astropy.io import fits


def _read_meta_vector(hdf5, key, n_epochs):
    if key not in hdf5:
        return np.full((n_epochs,), np.nan)

    values = np.asarray(hdf5[key][:]).reshape(-1)
    if len(values) != n_epochs:
        raise ValueError(
            "Dataset '{}' must have length {}, got {}.".format(
                key, n_epochs, len(values)
            )
        )
    return values


def _flatten_and_sort_epoch(wl2d, fl2d, sigma2d):
    """Flatten one (n_orders, n_pix) epoch into wavelength-sorted 1D arrays."""
    n_orders, n_pix = wl2d.shape
    order_index = np.repeat(np.arange(n_orders, dtype=np.int32), n_pix)
    pix_index = np.tile(np.arange(n_pix, dtype=np.int32), n_orders)

    wl1d = wl2d.reshape(-1)
    fl1d = fl2d.reshape(-1)
    sigma1d = sigma2d.reshape(-1)

    finite = np.isfinite(wl1d) & np.isfinite(fl1d) & np.isfinite(sigma1d)
    wl1d = wl1d[finite]
    fl1d = fl1d[finite]
    sigma1d = sigma1d[finite]
    order_index = order_index[finite]
    pix_index = pix_index[finite]

    sort_index = np.argsort(wl1d)
    return (
        wl1d[sort_index],
        fl1d[sort_index],
        sigma1d[sort_index],
        order_index[sort_index],
        pix_index[sort_index],
    )


def _build_hdus(wl1d, fl1d, sigma1d, order_index, pix_index, header):
    """Build the output HDU list with primary metadata and a SPECTRUM table."""
    primary_hdu = fits.PrimaryHDU(header=header)
    table_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="WAVELENGTH", format="D", array=wl1d),
            fits.Column(name="FLUX", format="D", array=fl1d),
            fits.Column(name="SIGMA", format="D", array=sigma1d),
            fits.Column(name="ORDER", format="J", array=order_index),
            fits.Column(name="PIXEL", format="J", array=pix_index),
        ],
        name="SPECTRUM",
    )
    return fits.HDUList([primary_hdu, table_hdu])


def convert_hdf5_to_fits(input_hdf5, output_dir, prefix=None, overwrite=False):
    """Convert a PSOAP HDF5 cube into one 1D FITS spectrum file per epoch."""
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with h5py.File(input_hdf5, "r") as hdf5:
        wl = hdf5["wl"][:]
        fl = hdf5["fl"][:]
        sigma = hdf5["sigma"][:]

        if wl.ndim != 3:
            raise ValueError(
                "Dataset 'wl' must have shape (n_epochs, n_orders, n_pix)."
            )
        if fl.shape != wl.shape or sigma.shape != wl.shape:
            raise ValueError("Datasets 'fl' and 'sigma' must match shape of 'wl'.")

        n_epochs = wl.shape[0]
        jd = _read_meta_vector(hdf5, "JD", n_epochs)
        bcv = _read_meta_vector(hdf5, "BCV", n_epochs)

    n_epochs, n_orders, n_pix = wl.shape

    if prefix is None:
        prefix = os.path.splitext(os.path.basename(input_hdf5))[0]

    for epoch in range(n_epochs):
        wl1d, fl1d, sigma1d, order_index, pix_index = _flatten_and_sort_epoch(
            wl[epoch], fl[epoch], sigma[epoch]
        )

        header = fits.Header()
        header["SRCFILE"] = os.path.basename(input_hdf5)
        header["EPOCH"] = int(epoch)
        header["JD"] = float(jd[epoch])
        header["BCV"] = float(bcv[epoch])
        header["NORDER"] = int(n_orders)
        header["NPIX"] = int(n_pix)
        header["NPTS"] = int(len(wl1d))

        hdul = _build_hdus(wl1d, fl1d, sigma1d, order_index, pix_index, header)

        output_name = "{}_epoch{:03d}.fits".format(prefix, epoch)
        output_path = os.path.join(output_dir, output_name)
        hdul.writeto(output_path, overwrite=overwrite)

    return n_epochs


def main():
    parser = argparse.ArgumentParser(
        description="Convert PSOAP HDF5 echelle spectra to per-epoch 1D FITS files."
    )
    parser.add_argument("input_hdf5", help="Input PSOAP-format HDF5 file.")
    parser.add_argument(
        "output_dir", help="Directory where per-epoch FITS files will be written."
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix (default: input file stem).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output FITS files if they already exist.",
    )
    args = parser.parse_args()

    n_epochs = convert_hdf5_to_fits(
        args.input_hdf5, args.output_dir, prefix=args.prefix, overwrite=args.overwrite
    )
    print("Wrote {} FITS files to {}".format(n_epochs, args.output_dir))


if __name__ == "__main__":
    main()
