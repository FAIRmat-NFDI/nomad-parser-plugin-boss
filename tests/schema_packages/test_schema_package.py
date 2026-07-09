import json
import re
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
    sanitize_h5_name,
    slice_group_names,
)

# Unique names: needed by the map/metamorphic tests, whose `names.index(...)`
# oracle is ambiguous under duplicates.
parameter_names = st.lists(
    st.text(min_size=1, max_size=12), min_size=2, max_size=8, unique=True
)
# Names that may repeat: for the group-name uniqueness property, which must
# hold even when parameters share a name.
parameter_names_maybe_duplicated = st.lists(
    st.text(min_size=1, max_size=12), min_size=2, max_size=8
)


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('phi', 'phi'),
        ('a/b', 'a_b'),  # path separator
        ('a#b', 'a_b'),  # reference-fragment marker
        ('spin  up', 'spin_up'),  # whitespace run
        (' theta ', 'theta'),  # surrounding whitespace stripped
        ('/', ''),  # no usable characters
        ('   ', ''),
    ],
)
def test_sanitize_h5_name(raw, expected):
    assert sanitize_h5_name(raw) == expected


@pytest.mark.parametrize(
    ('names', 'expected'),
    [
        # named after the compared parameters, in generate_slices order
        (['a', 'b', 'c'], ['a_vs_b', 'a_vs_c', 'b_vs_c']),
        # a name that sanitizes to empty falls back to the indexed slice name
        (['/', 'b'], ['slice_0']),
        # names sanitizing to the same component collide -> disambiguated
        (['x/y', 'x#y', 'z'], ['x_y_vs_x_y', 'x_y_vs_z_1', 'x_y_vs_z_2']),
        # duplicate parameter names collide the same way and stay unique
        (['a', 'a', 'b'], ['a_vs_a', 'a_vs_b_1', 'a_vs_b_2']),
    ],
)
def test_slice_group_names(names, expected):
    assert slice_group_names(names) == expected


def test_h5web_attribute_map():
    attribute_map = h5web_attribute_map(['a', 'b', 'c'])

    # C(3, 2) = 3 slices, pairing follows generate_slices: (a,b), (a,c), (b,c)
    assert [key for key in attribute_map if key.count('/') == 1] == [
        '/a_vs_b',
        '/a_vs_c',
        '/b_vs_c',
    ]
    assert attribute_map['/a_vs_b']['title'] == 'a vs b'
    assert attribute_map['/a_vs_c']['title'] == 'a vs c'
    assert attribute_map['/b_vs_c/parameters_x']['long_name'] == 'b'
    assert attribute_map['/b_vs_c/parameters_y']['long_name'] == 'c'

    # The fixed NeXus constants are asserted here, example-based, exactly once
    group_attrs = attribute_map['/a_vs_b']
    assert group_attrs['NX_class'] == 'NXdata'
    assert group_attrs['signal'] == 'fit'
    assert group_attrs['axes'] == ['iteration', 'parameters_x', 'parameters_y']
    assert group_attrs['auxiliary_signals'] == ['uncertainty']
    assert attribute_map['/a_vs_b/fit']['units'] == 'eV'
    assert attribute_map['/a_vs_b/uncertainty']['units'] == 'eV'


@given(names=parameter_names_maybe_duplicated)
def test_slice_group_names_properties(names):
    """
    Hypothesis property test for `slice_group_names` over arbitrary lists of
    2-8 names, duplicates allowed. The count (C(n,2)) is structural and pinned
    by the example tests; generation targets the two invariants that adversarial
    input can break, including parameters that share a name:

      G2: every group name is HDF5-path-safe -- non-empty and free of '/', '#',
          and whitespace, since it is used directly as a path component.
      G3: the group names are unique, even when sanitization collisions or
          duplicate parameter names would otherwise produce the same name.
    """
    group_names = slice_group_names(names)

    for group_name in group_names:
        assert group_name  # G2
        assert not re.search(r'[\s/#]', group_name)  # G2
    assert len(set(group_names)) == len(group_names)  # G3


@given(names=parameter_names)
def test_h5web_attribute_map_properties(names):
    """
    Hypothesis property test over arbitrary lists of 2-8 unique names.
    Only input-dependent structure is asserted here; the fixed NeXus
    constants are covered once in the example-based test above.

    Predicates, for `names` of length n:
      P1: the map contains exactly C(n,2) group paths, one per slice group
          from `slice_group_names`, in that order.
      P2: the (parameters_x, parameters_y) long_name pairs of the groups
          enumerate every index pair (i, j) with i < j exactly once, in
          generate_slices order — checked against the independent oracle
          itertools.combinations. long_names carry the original (unsanitized)
          names, so the lookup is exact.
      P3 (relational): each group's title agrees with its own axis
          long_names: title == f'{x} vs {y}'.
      P4: every group carries attribute entries for all five datasets
          (fit, uncertainty, iteration, parameters_x, parameters_y).
    """
    attribute_map = h5web_attribute_map(names)

    groups = [key for key in attribute_map if key.count('/') == 1]
    n = len(names)
    assert groups == [f'/{group}' for group in slice_group_names(names)]  # P1

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
    input names must transform exactly the name-derived values (group paths,
    axis long_names, titles) and leave every other attribute untouched.

    The rename appends '~renamed' to each name. Because group paths now
    encode the parameter names, the two maps have different keys but the same
    structure, so entries are compared position-by-position.

    Predicates, comparing map(names) with map(renamed):
      M1: both maps have the same number of entries and each aligned pair
          sits at the same structural depth (group vs dataset).
      M2: axis dataset long_names transform by exactly the rename mapping.
      M3: group titles remain consistent with their own (renamed) axis
          long_names; all other group attributes are unchanged.
      M4: attributes of non-axis datasets are unchanged.
    """
    renamed = [f'{name}~renamed' for name in names]
    mapping = dict(zip(names, renamed))
    original = h5web_attribute_map(names)
    transformed = h5web_attribute_map(renamed)

    assert len(original) == len(transformed)  # M1
    for (path, attrs), (renamed_path, renamed_attrs) in zip(
        original.items(), transformed.items()
    ):
        assert path.count('/') == renamed_path.count('/')  # M1
        assert attrs.keys() == renamed_attrs.keys()
        if path.count('/') == 1:  # group node
            x_name = transformed[f'{renamed_path}/parameters_x']['long_name']
            y_name = transformed[f'{renamed_path}/parameters_y']['long_name']
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
    assert fit_reference.rsplit('#', 1)[-1] == '/x_vs_y/fit'

    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        assert_h5web_group(
            h5file,
            '/x_vs_y',
            title='x vs y',
            long_names={'parameters_x': 'x', 'parameters_y': 'y'},
        )


def test_eln_edit_renames_group_and_refreshes_labels(
    upload_dir, synthetic_pes, assert_h5web_group, monkeypatch
):
    """
    Replicate the editable-archive flow: a GUI edit rewrites the .archive.json
    and re-runs normalization (no parser). The slice group is renamed after the
    edited parameter_names and the labels follow, without recomputing the fits:
    the group is moved (data preserved) and the HDF5References are rewritten.
    """
    archive = parse(str(upload_dir / 'test.archive.yaml'))[0]
    normalize_all(archive)

    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        fit_before = h5file['/x_vs_y/fit'][()]

    edited_data = archive.data.m_to_dict(with_root_def=True)
    edited_data['parameter_names'] = ['alpha', 'beta']
    edited_file = upload_dir / 'boss.archive.json'
    edited_file.write_text(json.dumps({'data': edited_data}))

    def fail_compute(self, archive, logger):
        pytest.fail('The expensive BOSS compute must not run on an ELN edit')

    monkeypatch.setattr(ELNBOSSAnalysis, '_compute_pes', fail_compute)

    edited_archive = parse(str(edited_file))[0]
    normalize_all(edited_archive)

    # the HDF5Reference now points at the renamed group
    fit_reference = edited_archive.data.parameter_slices[0].fit
    assert fit_reference.rsplit('#', 1)[-1] == '/alpha_vs_beta/fit'

    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        assert '/x_vs_y' not in h5file  # group was moved, not copied
        assert_h5web_group(
            h5file,
            '/alpha_vs_beta',
            title='alpha vs beta',
            long_names={'parameters_x': 'alpha', 'parameters_y': 'beta'},
        )
        assert np.array_equal(h5file['/alpha_vs_beta/fit'][()], fit_before)


def test_eln_edit_permutation_preserves_data(upload_dir, synthetic_pes, monkeypatch):
    """
    A name edit that permutes the pairwise group names (e.g. swapping the labels
    of two parameters) makes group names cycle. The rename must move the groups
    consistently, so every slice still resolves to its own data rather than
    another slice's. On the direct-move code this cycle silently swapped the data.
    """
    mainfile = upload_dir / 'three.archive.yaml'
    mainfile.write_text(
        'data:\n'
        '  m_def: nomad_parser_plugin_boss.schema_packages.schema_package'
        '.ELNBOSSAnalysis\n'
        '  data_file: boss.rst\n'
        '  parameter_names:\n    - a\n    - b\n    - c\n'
    )
    archive = parse(str(mainfile))[0]
    normalize_all(archive)

    # the synthetic _compute_pes writes float(counter) into each slice's fit;
    # resolve each slice's value through its own reference
    def fit_value(parameter_slice, h5file):
        group = parameter_slice.fit.rsplit('#', 1)[1].rsplit('/', 1)[0].lstrip('/')
        return h5file[f'{group}/fit'][0, 0, 0]

    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        before = [fit_value(s, h5file) for s in archive.data.parameter_slices]
    assert before == [0.0, 1.0, 2.0]

    # swap the labels of parameters 1 and 2: a_vs_b <-> a_vs_c cycle
    edited = archive.data.m_to_dict(with_root_def=True)
    edited['parameter_names'] = ['a', 'c', 'b']
    edited_file = upload_dir / 'three.archive.json'
    edited_file.write_text(json.dumps({'data': edited}))

    def fail_compute(self, archive, logger):
        pytest.fail('The expensive BOSS compute must not run on an ELN edit')

    monkeypatch.setattr(ELNBOSSAnalysis, '_compute_pes', fail_compute)

    edited_archive = parse(str(edited_file))[0]
    normalize_all(edited_archive)

    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        after = [fit_value(s, h5file) for s in edited_archive.data.parameter_slices]
    assert after == before  # each slice kept its own data through the cycle


def test_rename_leaves_missing_group_reference_untouched(
    upload_dir, synthetic_pes, monkeypatch
):
    """
    If a slice's referenced group is absent from the auxiliary file, the rename
    must not point that reference at a destination it never created: the move is
    skipped and the reference is left as-is (with a warning).
    """
    archive = parse(str(upload_dir / 'test.archive.yaml'))[0]  # names x, y
    normalize_all(archive)

    # drop the group from the file, leaving the archive reference dangling
    with h5py.File(upload_dir / 'boss.h5', 'r+') as h5file:
        del h5file['x_vs_y']

    edited = archive.data.m_to_dict(with_root_def=True)
    edited['parameter_names'] = ['alpha', 'beta']
    edited_file = upload_dir / 'boss.archive.json'
    edited_file.write_text(json.dumps({'data': edited}))

    def fail_compute(self, archive, logger):
        pytest.fail('The expensive BOSS compute must not run on an ELN edit')

    monkeypatch.setattr(ELNBOSSAnalysis, '_compute_pes', fail_compute)

    edited_archive = parse(str(edited_file))[0]
    normalize_all(edited_archive)

    # the reference was not rewritten to the non-existent /alpha_vs_beta group
    fit_reference = edited_archive.data.parameter_slices[0].fit
    assert fit_reference.rsplit('#', 1)[-1] == '/x_vs_y/fit'


def test_recompute_clears_stale_groups(upload_dir, synthetic_pes):
    """
    A recomputation over an existing .h5 (e.g. after the analysis entry lost
    its computed slices) must rebuild a clean file, not leave the previously
    named group behind.
    """
    archive = parse(str(upload_dir / 'test.archive.yaml'))[0]  # names x, y
    normalize_all(archive)
    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        assert set(h5file.keys()) == {'x_vs_y'}

    # the archive file lost its slices and carries different names, forcing a
    # recomputation while the old .h5 (with the /x_vs_y group) is still present
    edited = archive.data.m_to_dict(with_root_def=True)
    edited['parameter_names'] = ['alpha', 'beta']
    edited['parameter_slices'] = []
    edited_file = upload_dir / 'boss.archive.json'
    edited_file.write_text(json.dumps({'data': edited}))

    edited_archive = parse(str(edited_file))[0]
    normalize_all(edited_archive)

    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        assert set(h5file.keys()) == {'alpha_vs_beta'}


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
            '/phi_vs_psi',
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

    # the mismatch aborts the refresh, so the group keeps its original name
    with h5py.File(upload_dir / 'boss.h5', 'r') as h5file:
        assert_h5web_group(
            h5file,
            '/x_vs_y',
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
        assert h5file['/x_vs_y/fit'].shape[1:] == (50, 50)
        assert h5file['/x_vs_y/fit'].shape[0] == h5file['/x_vs_y/iteration'].shape[0]
        assert_h5web_group(
            h5file,
            '/x_vs_y',
            title='x vs y',
            long_names={'parameters_x': 'x', 'parameters_y': 'y'},
        )
