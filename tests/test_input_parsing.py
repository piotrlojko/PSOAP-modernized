from pathlib import Path

import numpy as np
import pytest

from psoap.input_parsing import (
    load_spectrum_array,
    parse_spectra_list,
    render_model_config,
)


def test_parse_spectra_list_with_header(tmp_path, capsys):
    path = tmp_path / "spectra_list.txt"
    path.write_text(
        "filename date\n"
        "spec1.txt 2450000.0\n"
        "spec2.txt 2450001.5\n",
        encoding="utf-8",
    )
    filenames, dates = parse_spectra_list(str(path))
    assert filenames == ["spec1.txt", "spec2.txt"]
    assert np.allclose(dates, [2450000.0, 2450001.5])
    assert "Warning:" not in capsys.readouterr().out


def test_parse_spectra_list_without_header_warns(tmp_path, capsys):
    path = tmp_path / "spectra_list.txt"
    path.write_text(
        "spec1.txt 2450000.0\n"
        "spec2.txt 2450001.5\n",
        encoding="utf-8",
    )
    filenames, dates = parse_spectra_list(str(path))
    assert filenames == ["spec1.txt", "spec2.txt"]
    assert np.allclose(dates, [2450000.0, 2450001.5])
    assert "missing the expected header line" in capsys.readouterr().out


def test_parse_spectra_list_bad_column_count_fails(tmp_path):
    path = tmp_path / "spectra_list.txt"
    path.write_text("filename date\nspec1.txt 2450000.0 extra\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected 2 columns"):
        parse_spectra_list(str(path))


def test_parse_spectra_list_bad_date_fails(tmp_path):
    path = tmp_path / "spectra_list.txt"
    path.write_text("filename date\nspec1.txt not_a_date\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot be converted to float"):
        parse_spectra_list(str(path))


def test_load_spectrum_array_with_header(tmp_path, capsys):
    path = tmp_path / "spec.txt"
    path.write_text(
        "wavelength_Angstrom flux flux_err\n"
        "5000.0 1.0 0.01\n"
        "5001.0 0.9 0.02\n",
        encoding="utf-8",
    )
    arr = load_spectrum_array(str(path))
    assert arr.shape == (2, 3)
    assert np.allclose(arr[:, 0], [5000.0, 5001.0])
    assert "Warning:" not in capsys.readouterr().out


def test_load_spectrum_array_without_header_warns(tmp_path, capsys):
    path = tmp_path / "spec.txt"
    path.write_text("5000.0 1.0 0.01\n5001.0 0.9 0.02\n", encoding="utf-8")
    arr = load_spectrum_array(str(path))
    assert arr.shape == (2, 3)
    assert "missing the expected header line" in capsys.readouterr().out


def test_load_spectrum_array_bad_column_count_fails(tmp_path):
    path = tmp_path / "spec.txt"
    path.write_text("wavelength_Angstrom flux flux_err\n5000.0 1.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected 3 columns"):
        load_spectrum_array(str(path))


def test_load_spectrum_array_non_numeric_fails(tmp_path):
    path = tmp_path / "spec.txt"
    path.write_text(
        "wavelength_Angstrom flux flux_err\n5000.0 not_a_flux 0.01\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="contains non-numeric values"):
        load_spectrum_array(str(path))


def test_render_model_config_contains_model_and_yaml():
    config = {
        "model": "SB2",
        "spectra_list": "spectra_list.txt",
        "parameters": {"q": 0.2},
    }
    text = render_model_config(config)
    assert "Current model configuration (SB2)" in text
    assert "spectra_list: spectra_list.txt" in text
    assert "parameters:" in text
