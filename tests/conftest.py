import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
import structlog
from nomad.utils import structlogging
from nomad_measurements.utils import Dataset, HDF5Handler
from structlog.testing import LogCapture

from nomad_parser_plugin_boss.schema_packages.schema_package import (
    ELNBOSSAnalysis,
    ParameterSpaceSlice,
    generate_slices,
    h5web_attribute_map,
)

DATA_DIR = Path(__file__).parent / 'data'


@pytest.fixture(autouse=True, scope='session')
def _short_log_format():
    """
    Render nomad's structlog output in short form during the test session,
    restoring the global logging state on teardown so the patch cannot leak
    into other suites and make test ordering matter.
    """
    original_formatter = logging.Formatter
    original_short_format = structlogging.ConsoleFormatter.short_format
    structlogging.ConsoleFormatter.short_format = True
    logging.Formatter = structlogging.ConsoleFormatter
    try:
        yield
    finally:
        logging.Formatter = original_formatter
        structlogging.ConsoleFormatter.short_format = original_short_format


@pytest.fixture(name='caplog', scope='function')
def fixture_caplog():
    """Capture structlog entries for assertions on log output."""
    caplog = LogCapture()
    processors = structlog.get_config()['processors']
    old_processors = processors.copy()
    try:
        processors.clear()
        processors.append(caplog)
        structlog.configure(processors=processors)
        yield caplog
    finally:
        processors.clear()
        processors.extend(old_processors)
        structlog.configure(processors=processors)


@pytest.fixture
def upload_dir(tmp_path):
    """
    Isolated stand-in for an upload's raw directory, seeded with the test data
    files. `nomad.client.parse` roots its context at the mainfile's directory,
    so generated .h5/.archive.json files land here and are cleaned up with it.
    """
    for file in DATA_DIR.iterdir():
        shutil.copy(file, tmp_path / file.name)
    return tmp_path


def synthetic_compute_pes(self, archive, logger):
    """
    BOSS-free stand-in for `ELNBOSSAnalysis._compute_pes`: writes small
    deterministic arrays through the real HDF5Handler machinery, exercising
    dataset creation, NeXus attributes, and HDF5Reference population.
    """
    if not self.parameter_names:
        self.parameter_names = ['parameter_0', 'parameter_1']
    self.auxiliary_file = f'{self.data_file.rsplit(".", 1)[0]}.h5'
    handler = HDF5Handler(filename=self.auxiliary_file, archive=archive, logger=logger)

    iterations = np.arange(2, 0, -1)
    grid = np.linspace(0.0, 1.0, num=5)
    for counter, _ in enumerate(generate_slices(len(self.parameter_names))):
        self.parameter_slices.append(ParameterSpaceSlice())
        values = np.full((len(iterations), 5, 5), float(counter))
        for name, data in (
            ('fit', values),
            ('uncertainty', values / 10.0),
            ('iteration', iterations),
            ('parameters_x', grid),
            ('parameters_y', grid),
        ):
            handler.add_dataset(
                path=f'/slice_{counter}/{name}',
                dataset=Dataset(
                    data=data,
                    archive_path=f'data.parameter_slices[{counter}].{name}',
                ),
            )

    for path, attributes in h5web_attribute_map(self.parameter_names).items():
        handler.add_attribute(path=path, params=attributes)
    handler.write_file()
    self.figures = []


@pytest.fixture
def synthetic_pes(monkeypatch):
    """Replace the expensive BOSS model reconstruction with the synthetic stub."""
    monkeypatch.setattr(ELNBOSSAnalysis, '_compute_pes', synthetic_compute_pes)


@pytest.fixture
def assert_h5web_group():
    """
    Assert that an h5py group carries the NeXus attributes H5Web renders from:
    signal/axes/auxiliary_signals on the group, long_name on the axis datasets.
    """

    def _assert(  # noqa: PLR0913
        h5file,
        group_path,
        signal='fit',
        axes=('iteration', 'parameters_x', 'parameters_y'),
        auxiliary_signals=('uncertainty',),
        title=None,
        long_names=None,
    ):
        group = h5file[group_path]
        assert group.attrs['NX_class'] == 'NXdata'
        assert group.attrs['signal'] == signal
        assert list(group.attrs['axes']) == list(axes)
        assert list(group.attrs['auxiliary_signals']) == list(auxiliary_signals)
        if title is not None:
            assert group.attrs['title'] == title
        assert signal in group, f'signal dataset "{signal}" missing in {group_path}'
        for axis in axes:
            assert axis in group, f'axis dataset "{axis}" missing in {group_path}'
        for name, long_name in (long_names or {}).items():
            assert group[name].attrs['long_name'] == long_name

    return _assert
