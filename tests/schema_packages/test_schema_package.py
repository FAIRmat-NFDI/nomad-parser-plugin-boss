import json
from itertools import combinations

import h5py
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from nomad.client import normalize_all, parse

from nomad_parser_plugin_boss.schema_packages.schema_package import (
    ELNBOSSAnalysis,
    h5web_attribute_map,
)

parameter_names = st.lists(
    st.text(min_size=1, max_size=12), min_size=2, max_size=8, unique=True
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

    # The fixed NeXus constants are asserted here, example-based, exactly once
    group_attrs = attribute_map['/slice_0']
    assert group_attrs['NX_class'] == 'NXdata'
    assert group_attrs['signal'] == 'fit'
    assert group_attrs['axes'] == ['iteration', 'parameters_x', 'parameters_y']
    assert group_attrs['auxiliary_signals'] == ['uncertainty']
    assert attribute_map['/slice_0/fit']['units'] == 'eV'
    assert attribute_map['/slice_0/uncertainty']['units'] == 'eV'


@given(names=parameter_names)
def test_h5web_attribute_map_properties(names):
    """
    Hypothesis property test over arbitrary lists of 2-8 unique names.
    Only input-dependent structure is asserted here; the fixed NeXus
    constants are covered once in the example-based test above.

    Predicates, for `names` of length n:
      P1: the map contains exactly C(n,2) group paths, named
          /slice_0 .. /slice_{C(n,2)-1} in that order.
      P2: the (parameters_x, parameters_y) long_name pairs of the groups
          enumerate every index pair (i, j) with i < j exactly once, in
          generate_slices order — checked against the independent oracle
          itertools.combinations.
      P3 (relational): each group's title agrees with its own axis
          long_names: title == f'{x} vs {y}'.
      P4: every group carries attribute entries for all five datasets
          (fit, uncertainty, iteration, parameters_x, parameters_y).
    """
    attribute_map = h5web_attribute_map(names)

    groups = [key for key in attribute_map if key.count('/') == 1]
    n = len(names)
    assert groups == [f'/slice_{i}' for i in range(n * (n - 1) // 2)]

    axis_pairs = []
    for group in groups:
        x_name = attribute_map[f'{group}/parameters_x']['long_name']
        y_name = attribute_map[f'{group}/parameters_y']['long_name']
        axis_pairs.append((names.index(x_name), names.index(y_name)))
        assert attribute_map[group]['title'] == f'{x_name} vs {y_name}'
        for dataset in ('fit', 'uncertainty', 'iteration', 'parameters_x',
                        'parameters_y'):
            assert f'{group}/{dataset}' in attribute_map

    assert axis_pairs == list(combinations(range(n), 2))


@given(names=parameter_names)
def test_h5web_attribute_map_rename_metamorphic(names):
    """
    Hypothesis metamorphic property: applying a bijective rename to the
    input names must change exactly the name-derived attribute values and
    nothing else.

    Predicates, comparing map(names) with map(renamed) where
    renamed[i] = names[i] + suffix (a bijection by construction):
      M1: the set of attribute paths is identical.
      M2: axis dataset long_names transform by exactly the rename mapping.
      M3: group titles remain consistent with their own (renamed) axis
          long_names; all other group attributes are unchanged.
      M4: attributes of non-axis datasets are unchanged.
    """
    renamed = [f'{name}~renamed' for name in names]
    mapping = dict(zip(names, renamed))
    original = h5web_attribute_map(names)
    transformed = h5web_attribute_map(renamed)

    assert original.keys() == transformed.keys()  # M1
    for path, attrs in original.items():
        renamed_attrs = transformed[path]
        assert attrs.keys() == renamed_attrs.keys()
        if path.count('/') == 1:  # group
            x_name = transformed[f'{path}/parameters_x']['long_name']
            y_name = transformed[f'{path}/parameters_y']['long_name']
            assert renamed_attrs['title'] == f'{x_name} vs {y_name}'  # M3
            assert {k: v for k, v in renamed_attrs.items() if k != 'title'} == {
                k: v for k, v in attrs.items() if k != 'title'
            }
        elif path.endswith(('/parameters_x', '/parameters_y')):
            assert renamed_attrs['long_name'] == mapping[attrs['long_name']]  # M2
        else:
            assert renamed_attrs == attrs  # M4


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


@pytest.mark.parametrize(
    'config_filename', ['boss_analysis.yml', 'boss_analysis.yaml']
)
def test_analysis_config_sets_names(
    upload_dir, synthetic_pes, assert_h5web_group, config_filename
):
    """A boss_analysis.yml next to the data file sets the names beforehand."""
    (upload_dir / config_filename).write_text('parameter_names:\n  - phi\n  - psi\n')
    mainfile = upload_dir / 'noname.archive.yaml'
    mainfile.write_text(
        'data:\n'
        '  m_def: nomad_parser_plugin_boss.schema_packages.schema_package'
        '.ELNBOSSAnalysis\n'
        '  data_file: boss.rst\n'
    )

    archive = parse(str(mainfile))[0]
    normalize_all(archive)

    assert list(archive.data.parameter_names) == ['phi', 'psi']
    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        assert_h5web_group(
            h5file,
            '/slice_0',
            title='phi vs psi',
            long_names={'parameters_x': 'phi', 'parameters_y': 'psi'},
        )


def test_analysis_config_fallback_on_corrupt_file(upload_dir, synthetic_pes):
    """A corrupt boss_analysis.yml must not prevent reading the .yaml one."""
    (upload_dir / 'boss_analysis.yml').write_text('parameter_names: [unclosed')
    (upload_dir / 'boss_analysis.yaml').write_text(
        'parameter_names:\n  - phi\n  - psi\n'
    )
    mainfile = upload_dir / 'noname.archive.yaml'
    mainfile.write_text(
        'data:\n'
        '  m_def: nomad_parser_plugin_boss.schema_packages.schema_package'
        '.ELNBOSSAnalysis\n'
        '  data_file: boss.rst\n'
    )

    archive = parse(str(mainfile))[0]
    normalize_all(archive)

    assert list(archive.data.parameter_names) == ['phi', 'psi']


def test_analysis_config_does_not_override_eln(upload_dir, synthetic_pes):
    """Names already set (e.g. in the ELN or archive file) win over the config."""
    (upload_dir / 'boss_analysis.yml').write_text(
        'parameter_names:\n  - phi\n  - psi\n'
    )

    archive = parse(str(upload_dir / 'test.archive.yaml'))[0]
    normalize_all(archive)

    assert list(archive.data.parameter_names) == ['x', 'y']


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
