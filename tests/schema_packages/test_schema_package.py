import json

import h5py
import numpy as np
import pytest
from nomad.client import normalize_all, parse

from nomad_parser_plugin_boss.schema_packages.schema_package import (
    ELNBOSSAnalysis,
    h5web_attribute_map,
)


def test_h5web_attribute_map():
    attribute_map = h5web_attribute_map(['a', 'b', 'c'])

    # C(3, 2) = 3 slices, pairing follows generate_slices: (a,b), (a,c), (b,c)
    assert [key for key in attribute_map if key.count('/') == 1] == [
        '/slice_0',
        '/slice_1',
        '/slice_2',
    ]
    assert attribute_map['/slice_0']['title'] == 'a vs b'
    assert attribute_map['/slice_1']['title'] == 'a vs c'
    assert attribute_map['/slice_2/parameters_x']['long_name'] == 'b'
    assert attribute_map['/slice_2/parameters_y']['long_name'] == 'c'

    group_attrs = attribute_map['/slice_0']
    assert group_attrs['NX_class'] == 'NXdata'
    assert group_attrs['signal'] == 'fit'
    assert group_attrs['auxiliary_signals'] == ['uncertainty']
    assert attribute_map['/slice_0/fit']['units'] == 'eV'
    assert attribute_map['/slice_0/uncertainty']['units'] == 'eV'


def test_first_pass_labels(upload_dir, synthetic_pes, assert_h5web_group):
    archive = parse(str(upload_dir / 'test.archive.yaml'))[0]
    normalize_all(archive)

    data = archive.data
    assert data.auxiliary_file == 'boss.h5'
    fit_reference = data.parameter_slices[0].fit
    assert fit_reference.startswith('/uploads/')
    assert fit_reference.rsplit('#', 1)[-1] == '/slice_0/fit'

    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        assert_h5web_group(
            h5file,
            '/slice_0',
            title='x vs y',
            long_names={'parameters_x': 'x', 'parameters_y': 'y'},
        )


def test_eln_edit_refreshes_labels(
    upload_dir, synthetic_pes, assert_h5web_group, monkeypatch
):
    """
    Replicate the editable-archive flow: a GUI edit rewrites the .archive.json
    and re-runs normalization (no parser). The labels in the auxiliary HDF5 file
    must follow the edited parameter_names without recomputing the fits.
    """
    archive = parse(str(upload_dir / 'test.archive.yaml'))[0]
    normalize_all(archive)

    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        fit_before = h5file['/slice_0/fit'][()]

    edited_data = archive.data.m_to_dict(with_root_def=True)
    edited_data['parameter_names'] = ['alpha', 'beta']
    edited_file = upload_dir / 'boss.archive.json'
    edited_file.write_text(json.dumps({'data': edited_data}))

    def fail_compute(self, archive, logger):
        pytest.fail('The expensive BOSS compute must not run on an ELN edit')

    monkeypatch.setattr(ELNBOSSAnalysis, '_compute_pes', fail_compute)

    edited_archive = parse(str(edited_file))[0]
    normalize_all(edited_archive)

    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        assert_h5web_group(
            h5file,
            '/slice_0',
            title='alpha vs beta',
            long_names={'parameters_x': 'alpha', 'parameters_y': 'beta'},
        )
        assert np.array_equal(h5file['/slice_0/fit'][()], fit_before)


def test_name_count_mismatch_warns(
    upload_dir, synthetic_pes, assert_h5web_group, caplog
):
    archive = parse(str(upload_dir / 'test.archive.yaml'))[0]
    normalize_all(archive)

    archive.data.parameter_names = ['a', 'b', 'c', 'd']
    normalize_all(archive)

    warnings = [
        record
        for record in caplog.entries
        if record['log_level'] == 'warning' and 'does not match' in record['event']
    ]
    assert warnings, 'expected a slice-count mismatch warning'

    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        assert_h5web_group(
            h5file,
            '/slice_0',
            title='x vs y',
            long_names={'parameters_x': 'x', 'parameters_y': 'y'},
        )


@pytest.mark.slow
def test_real_boss_compute(upload_dir, assert_h5web_group):
    """End-to-end with the real BOSS postprocessing on tests/data/boss.rst."""
    archive = parse(str(upload_dir / 'test.archive.yaml'))[0]
    normalize_all(archive)

    data = archive.data
    assert len(data.parameter_slices) == 1
    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        assert h5file['/slice_0/fit'].shape[1:] == (50, 50)
        assert h5file['/slice_0/fit'].shape[0] == h5file['/slice_0/iteration'].shape[0]
        assert_h5web_group(
            h5file,
            '/slice_0',
            title='x vs y',
            long_names={'parameters_x': 'x', 'parameters_y': 'y'},
        )
