"""
Input parsing helpers for sampler workflows.
"""

import logging

import numpy as np
import yaml


_SPECTRA_LIST_KEYS = ("filename", "date")
_SPECTRUM_KEYS = ("wavelength_Angstrom", "flux", "flux_err")


def _iter_noncomment_lines(path):
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            yield line_no, stripped


def _warn(message):
    print("Warning:", message)
    logging.warning(message)


def parse_spectra_list(path):
    rows = list(_iter_noncomment_lines(path))
    if not rows:
        raise RuntimeError(
            "Failed to parse spectra list '{}': file has no non-comment content."
            .format(path)
        )

    first_line_no, first_line = rows[0]
    first_fields = first_line.split()
    has_header = tuple(first_fields) == _SPECTRA_LIST_KEYS

    if has_header:
        data_rows = rows[1:]
    else:
        _warn(
            "Spectra list '{}' is missing the expected header line '{} {}'; "
            "proceeding with assumed columns by order."
            .format(path, *_SPECTRA_LIST_KEYS)
        )
        data_rows = rows

    if not data_rows:
        raise RuntimeError(
            "Failed to parse spectra list '{}': no data rows found after header at line {}."
            .format(path, first_line_no)
        )

    filenames = []
    dates = []
    for line_no, row in data_rows:
        fields = row.split()
        if len(fields) != 2:
            raise RuntimeError(
                "Failed to parse spectra list '{}': expected 2 columns at line {}, got {}."
                .format(path, line_no, len(fields))
            )
        filenames.append(fields[0])
        try:
            dates.append(float(fields[1]))
        except ValueError as exc:
            raise RuntimeError(
                "Failed to parse spectra list '{}': date column at line {} "
                "cannot be converted to float."
                .format(path, line_no)
            ) from exc

    return filenames, np.asarray(dates, dtype=np.float64)


def load_spectrum_array(path):
    rows = list(_iter_noncomment_lines(path))
    if not rows:
        raise RuntimeError(
            "Failed to parse spectrum '{}': file has no non-comment content."
            .format(path)
        )

    first_fields = rows[0][1].split()
    has_header = tuple(first_fields) == _SPECTRUM_KEYS

    if has_header:
        data_rows = rows[1:]
    else:
        _warn(
            "Spectrum '{}' is missing the expected header line '{} {} {}'; "
            "proceeding with assumed columns by order."
            .format(path, *_SPECTRUM_KEYS)
        )
        data_rows = rows

    if not data_rows:
        raise RuntimeError(
            "Failed to parse spectrum '{}': no data rows found."
            .format(path)
        )

    data = np.empty((len(data_rows), 3), dtype=np.float64)
    for i, (line_no, row) in enumerate(data_rows):
        fields = row.split()
        if len(fields) != 3:
            raise RuntimeError(
                "Failed to parse spectrum '{}': expected 3 columns at line {}, got {}."
                .format(path, line_no, len(fields))
            )
        try:
            data[i, 0] = float(fields[0])
            data[i, 1] = float(fields[1])
            data[i, 2] = float(fields[2])
        except ValueError as exc:
            raise RuntimeError(
                "Failed to parse spectrum '{}': line {} contains non-numeric values."
                .format(path, line_no)
            ) from exc

    return data


def render_model_config(config):
    model = config.get("model", "<unknown>")
    body = yaml.safe_dump(config, sort_keys=False, default_flow_style=False).strip()
    return "\nCurrent model configuration ({})\n{}\n{}\n".format(
        model, "-" * 60, body
    )


def print_and_log_model_config(config):
    text = render_model_config(config)
    print(text)
    logging.info("\n%s", text)
