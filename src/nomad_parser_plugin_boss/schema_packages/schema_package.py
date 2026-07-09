from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
from nomad.datamodel.data import ArchiveSection, EntryData, Schema
from nomad.datamodel.hdf5 import HDF5Reference
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    ELNComponentEnum,
    H5WebAnnotation,
    SectionProperties,
)
from nomad.datamodel.metainfo.plot import PlotSection
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection
from nomad_measurements.utils import Dataset, HDF5Handler

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger


m_package = SchemaPackage()


def generate_slices(ranks: int) -> Generator:
    """Produce all possible index pairs defining slices of the parameter space."""
    for main_rank in range(ranks):
        for upper_rank in range(main_rank + 1, ranks):
            yield main_rank, upper_rank


def h5web_attribute_map(parameter_names: list[str]) -> dict[str, dict]:
    """
    Map HDF5 paths to the NeXus attributes that H5Web reads for rendering,
    covering all parameter-space slices for the given parameter names.
    Axis labels come from the `long_name` attribute of the axis datasets.
    """
    names = list(parameter_names)
    attribute_map = {}
    for counter, (main_rank, upper_rank) in enumerate(generate_slices(len(names))):
        prefix = f'/slice_{counter}'
        attribute_map[prefix] = dict(
            NX_class='NXdata',
            signal='fit',
            axes=['iteration', 'parameters_x', 'parameters_y'],
            auxiliary_signals=['uncertainty'],
            title=f'{names[main_rank]} vs {names[upper_rank]}',
        )
        attribute_map[f'{prefix}/parameters_x'] = dict(long_name=names[main_rank])
        attribute_map[f'{prefix}/parameters_y'] = dict(long_name=names[upper_rank])
        attribute_map[f'{prefix}/fit'] = dict(
            long_name='Potential Energy Surface Fit', units='eV'
        )
        attribute_map[f'{prefix}/uncertainty'] = dict(
            long_name='Fit Uncertainty', units='eV'
        )
        attribute_map[f'{prefix}/iteration'] = dict(long_name='Iteration')
    return attribute_map


class ParameterSpaceSlice(ArchiveSection):
    # ! TODO use `PhysicalProperty`
    # The section-level annotation is what the GUI uses to locate the plot;
    # label text (`long_name`, `title`) lives as attributes in the HDF5 file.
    m_def = Section(
        label='Parameter Space Slice',
        a_h5web=H5WebAnnotation(
            signal='fit',
            auxiliary_signals=['uncertainty'],
            axes=['iteration', 'parameters_x', 'parameters_y'],
        ),
    )

    fit = Quantity(
        type=HDF5Reference,
        unit='eV',
    )

    uncertainty = Quantity(
        type=HDF5Reference,
        unit='eV',
    )

    iteration = Quantity(
        type=HDF5Reference,
        description="""
        Iteration number of the data aggregation process in the fit.
        Iterations are numbered starting at 1 and stored in descending order.
        """,
    )

    parameters_x = Quantity(
        type=HDF5Reference,
    )

    parameters_y = Quantity(
        type=HDF5Reference,
    )


class PotentialEnergySurfaceFit(Schema):
    data_file = Quantity(
        type=str,
        description='Path to the BOSS .rst data file',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.FileEditQuantity,
        ),
    )

    auxiliary_file = Quantity(
        type=str,
        description='Name of the HDF5 auxiliary file containing the PES data',
    )

    parameter_names = Quantity(
        type=str,
        shape=['*'],
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )

    parameter_slices = SubSection(sub_section=ParameterSpaceSlice.m_def, repeats=True)

    # Optional per-upload configuration file, read from the data file's
    # directory. Kept generic so future options can be added without renaming.
    config_filenames = ('boss_analysis.yml', 'boss_analysis.yaml')

    def load_analysis_config(
        self, archive: 'EntryArchive', logger: 'BoundLogger'
    ) -> dict:
        """
        Read the optional `boss_analysis.yml` (or `.yaml`) configuration file
        placed next to the data file. Returns an empty dict if the file is
        missing or malformed. Currently supported keys: `parameter_names`.
        """
        import os

        import yaml

        directory = os.path.dirname(self.data_file or '')
        for filename in self.config_filenames:
            path = os.path.join(directory, filename) if directory else filename
            if not archive.m_context.raw_path_exists(path):
                continue
            try:
                with archive.m_context.raw_file(path) as file_handle:
                    content = yaml.safe_load(file_handle)
            except Exception as e:
                logger.warning(
                    'Could not read analysis config file. Trying next candidate.',
                    file=path,
                    error=str(e),
                )
                continue
            if isinstance(content, dict):
                logger.info('Loaded analysis config file.', file=path)
                return content
            logger.warning(
                'Invalid analysis config file: expected a mapping, e.g. a '
                '`parameter_names` key holding a list of names. '
                'Trying next candidate.',
                file=path,
            )
        return {}

    def refresh_h5web_labels(
        self, archive: 'EntryArchive', logger: 'BoundLogger'
    ) -> None:
        """
        Write the current `parameter_names` as NeXus attributes into the auxiliary
        HDF5 file. Runs on every normalization, so ELN edits of `parameter_names`
        update the H5Web axis labels without recomputing the fits.
        """
        if not self.parameter_names or not self.auxiliary_file:
            return
        if not archive.m_context.raw_path_exists(self.auxiliary_file):
            return

        n_params = len(self.parameter_names)
        expected_slices = n_params * (n_params - 1) // 2  # C(n, 2)
        n_slices = len(self.parameter_slices)
        if n_slices != expected_slices:
            logger.warning(
                'Number of slices does not match expected combinations. '
                'Not updating H5Web labels.',
                n_params=n_params,
                n_slices=n_slices,
                expected_slices=expected_slices,
            )
            return

        handler = HDF5Handler(
            filename=self.auxiliary_file, archive=archive, logger=logger
        )
        for path, attributes in h5web_attribute_map(self.parameter_names).items():
            handler.add_attribute(path=path, params=attributes)
        handler.write_file()

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        super().normalize(archive, logger)
        self.refresh_h5web_labels(archive, logger)


class ELNBOSSAnalysis(PotentialEnergySurfaceFit, EntryData, PlotSection):
    """
    ELN entry for BOSS Bayesian Optimization analysis results.
    This section combines the data model with ELN features and plotting capabilities.
    """

    m_def = Section(
        label='BOSS Analysis',
        a_eln=ELNAnnotation(
            lane_width='800px',
            properties=SectionProperties(
                order=[
                    'name',
                    'data_file',
                    'parameter_names',
                    'description',
                ]
            ),
        ),
        a_h5web=H5WebAnnotation(paths=['parameter_slices/0']),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        """
        Normalization method for the ELN entry. The expensive BOSS model
        reconstruction only runs when its results are missing; the H5Web label
        refresh (via the parent class) runs on every normalization.
        """
        import os

        if self.data_file and not archive.metadata.entry_name:
            file_base = os.path.basename(self.data_file)
            archive.metadata.entry_name = f'BOSS Analysis: {file_base}'

        # Names set beforehand via the optional boss_analysis.yml config file;
        # never overrides names already set (e.g. edited in the ELN)
        if self.data_file and not self.parameter_names:
            names = self.load_analysis_config(archive, logger).get('parameter_names')
            if names is not None and not (
                isinstance(names, list) and all(isinstance(n, str) for n in names)
            ):
                logger.warning(
                    'Invalid `parameter_names` in analysis config: expected a '
                    'list of names.',
                )
                names = None
            self.parameter_names = names or []

        needs_compute = self.data_file and (
            not self.parameter_slices
            or not self.auxiliary_file
            or not archive.m_context.raw_path_exists(self.auxiliary_file)
        )
        if needs_compute:
            self.parameter_slices = []
            try:
                self._compute_pes(archive, logger)
            except Exception as e:
                logger.error('Error parsing BOSS data', error=str(e), exc_info=True)
                raise

        super().normalize(archive, logger)

    def _compute_pes(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        """
        Parse the BOSS data file, reconstruct the PES fit and uncertainty on all
        2D parameter-space slices, and write them to the auxiliary HDF5 file.
        """
        import os

        from boss.bo.results import BOResults

        # Resolve via raw_file so it works in both server and client contexts
        with archive.m_context.raw_file(self.data_file) as data_file_handle:
            mainfile = data_file_handle.name
        boss_out = os.path.join(os.path.dirname(mainfile), 'boss.out')

        res = BOResults.from_file(mainfile, boss_out)

        iter_no = res.settings.get('iterpts', 1)
        no_grid_points = 50  # Can be made configurable

        bounds = res.settings.get('bounds', [])
        if bounds is None or len(bounds) == 0:
            logger.warning('No bounds found in BOSS results')
            return

        def compute_parameters(rank: int):
            return np.linspace(bounds[rank][0], bounds[rank][1], num=no_grid_points)

        # Default names; kept if the user already provided their own
        if self.parameter_names and len(self.parameter_names) != len(bounds):
            logger.warning(
                'Number of parameter names does not match the number of '
                'parameters. Using default names.',
                n_names=len(self.parameter_names),
                n_parameters=len(bounds),
            )
        if not self.parameter_names or len(self.parameter_names) != len(bounds):
            self.parameter_names = [f'parameter_{i}' for i in range(len(bounds))]

        h5_filename = f'{self.data_file.rsplit(".", 1)[0]}.h5'
        self.auxiliary_file = h5_filename
        handler = HDF5Handler(filename=h5_filename, archive=archive, logger=logger)

        iteration_procedure = np.arange(iter_no, 0, -1)

        # All parameters not in the slice are fixed to the global-minimum point
        x_default = np.atleast_2d(res.select('x_glmin', iter_no))

        for parameter_counter, rank in enumerate(generate_slices(len(bounds))):
            main_rank, upper_rank = rank
            mu_all_slices, var_all_slices = [], []

            # Query points on the 2D slice grid, built directly instead of via
            # PPMain/build_query_points, whose pp_model_slice indexing changed
            # between aalto-boss releases
            x_grid, y_grid = np.meshgrid(
                compute_parameters(main_rank), compute_parameters(upper_rank)
            )
            X = np.tile(x_default, (no_grid_points**2, 1))
            X[:, main_rank] = x_grid.ravel()
            X[:, upper_rank] = y_grid.ravel()

            for iteration in iteration_procedure:
                mu, var = res.reconstruct_model(iteration).predict(X)
                mu_all_slices.append(mu.reshape(no_grid_points, no_grid_points))
                var_all_slices.append(var.reshape(no_grid_points, no_grid_points))

            self.parameter_slices.append(ParameterSpaceSlice())

            handler.add_dataset(
                path=f'/slice_{parameter_counter}/fit',
                dataset=Dataset(
                    data=np.array(mu_all_slices),
                    archive_path=f'data.parameter_slices[{parameter_counter}].fit',
                ),
            )

            handler.add_dataset(
                path=f'/slice_{parameter_counter}/uncertainty',
                dataset=Dataset(
                    data=np.sqrt(np.array(var_all_slices)),
                    archive_path=(
                        f'data.parameter_slices[{parameter_counter}].uncertainty'
                    ),
                ),
            )

            handler.add_dataset(
                path=f'/slice_{parameter_counter}/iteration',
                dataset=Dataset(
                    data=iteration_procedure,
                    archive_path=(
                        f'data.parameter_slices[{parameter_counter}].iteration'
                    ),
                ),
            )

            handler.add_dataset(
                path=f'/slice_{parameter_counter}/parameters_x',
                dataset=Dataset(
                    data=np.array(compute_parameters(main_rank)),
                    archive_path=(
                        f'data.parameter_slices[{parameter_counter}].parameters_x'
                    ),
                ),
            )

            handler.add_dataset(
                path=f'/slice_{parameter_counter}/parameters_y',
                dataset=Dataset(
                    data=np.array(compute_parameters(upper_rank)),
                    archive_path=(
                        f'data.parameter_slices[{parameter_counter}].parameters_y'
                    ),
                ),
            )

        for path, attributes in h5web_attribute_map(self.parameter_names).items():
            handler.add_attribute(path=path, params=attributes)

        handler.write_file()

        # PlotSection expects figures to be initialized for the Overview tab
        self.figures = []

        logger.info(
            'Successfully parsed BOSS data and created HDF5 file',
            n_slices=len(self.parameter_slices),
            n_iterations=len(iteration_procedure),
            h5_file=h5_filename,
        )


class RawFileBOSSData(EntryData):
    """
    Entry section for a BOSS raw data file.
    Contains only a reference to the main ELN measurement entry.
    """

    measurement = Quantity(
        type=ELNBOSSAnalysis,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ReferenceEditQuantity,
        ),
    )


m_package.__init_metainfo__()
