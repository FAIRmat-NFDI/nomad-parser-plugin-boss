import re
from collections import Counter
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

# Datasets written per slice group, in the order H5Web expects them.
SLICE_DATASETS = ('fit', 'uncertainty', 'iteration', 'parameters_x', 'parameters_y')


def generate_slices(ranks: int) -> Generator:
    """Produce all possible index pairs defining slices of the parameter space."""
    for main_rank in range(ranks):
        for upper_rank in range(main_rank + 1, ranks):
            yield main_rank, upper_rank


def sanitize_h5_name(name: str) -> str:
    """
    Make a parameter name safe as an HDF5 path component: collapse the path
    separator ``/``, the reference-fragment marker ``#``, and whitespace runs
    into single underscores. May return ``''`` for a name with no usable
    characters (e.g. ``'/'`` or ``'   '``).
    """
    return re.sub(r'[\s/#]+', '_', name).strip('_')


def slice_group_names(parameter_names: list[str]) -> list[str]:
    """
    Names of the HDF5 groups holding each 2D parameter-space slice, one per
    ``(i, j)`` index pair from `generate_slices` and in that order. Each group
    is named ``'{x}_vs_{y}'`` after the compared parameters so the H5Web tree
    reflects what is plotted.

    The order of the two names is significant: ``x`` is the lower-index
    parameter, shown on the ``parameters_x`` axis, and ``y`` is the
    higher-index parameter, shown on ``parameters_y``. So ``'a_vs_b'`` and
    ``'b_vs_a'`` denote different axis assignments and only the former is
    produced (``generate_slices`` always yields ``i < j``).

    A slice whose names do not yield a usable component falls back to
    ``'slice_{index}'``, and any name that would still collide (e.g. from
    duplicate parameter names) is disambiguated, so the returned list is
    always unique.
    """
    names = list(parameter_names)
    proposed = []
    for index, (main_rank, upper_rank) in enumerate(generate_slices(len(names))):
        x = sanitize_h5_name(names[main_rank])
        y = sanitize_h5_name(names[upper_rank])
        proposed.append(f'{x}_vs_{y}' if x and y else f'slice_{index}')

    counts = Counter(proposed)
    used: set[str] = set()
    result: list[str] = []
    for index, name in enumerate(proposed):
        unique_name = f'{name}_{index}' if counts[name] > 1 else name
        while unique_name in used:
            unique_name = f'{unique_name}_dup'
        used.add(unique_name)
        result.append(unique_name)
    return result


def h5web_attribute_map(parameter_names: list[str]) -> dict[str, dict]:
    """
    Map HDF5 paths to the NeXus attributes that H5Web reads for rendering,
    covering all parameter-space slices for the given parameter names.
    Axis labels come from the `long_name` attribute of the axis datasets.
    """
    names = list(parameter_names)
    group_names = slice_group_names(names)
    attribute_map = {}
    for group_name, (main_rank, upper_rank) in zip(
        group_names, generate_slices(len(names))
    ):
        prefix = f'/{group_name}'
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
        Reconcile the auxiliary HDF5 file with the current `parameter_names`.
        Runs on every normalization, so ELN edits of `parameter_names` rename the
        slice groups after the compared parameters and update the H5Web axis
        labels without recomputing the fits.
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

        self._rename_slice_groups(
            archive, slice_group_names(self.parameter_names), logger
        )

        handler = HDF5Handler(
            filename=self.auxiliary_file, archive=archive, logger=logger
        )
        for path, attributes in h5web_attribute_map(self.parameter_names).items():
            handler.add_attribute(path=path, params=attributes)
        handler.write_file()

    def _rename_slice_groups(
        self, archive: 'EntryArchive', group_names: list[str], logger: 'BoundLogger'
    ) -> None:
        """
        Rename the HDF5 slice groups to `group_names` and rewrite the affected
        `HDF5Reference` values. The rename is a metadata-only `h5py` move (no
        data copy); groups whose name is already correct are left untouched.
        References are only rewritten for groups that are actually moved, so a
        source group missing from the file cannot leave a reference dangling.
        """
        import h5py

        # Plan the renames: current group -> (new group, its reference updates)
        plans: dict[str, tuple[str, list[tuple[ParameterSpaceSlice, str, str]]]] = {}
        for group_name, parameter_slice in zip(group_names, self.parameter_slices):
            reference = parameter_slice.fit
            if not reference or '#' not in reference:
                continue
            current_group = reference.split('#', 1)[1].rsplit('/', 1)[0]
            new_group = f'/{group_name}'
            if current_group == new_group:
                continue
            updates = []
            for dataset in SLICE_DATASETS:
                dataset_reference = getattr(parameter_slice, dataset)
                if dataset_reference and '#' in dataset_reference:
                    prefix = dataset_reference.split('#', 1)[0]
                    updates.append(
                        (parameter_slice, dataset, f'{prefix}#{new_group}/{dataset}')
                    )
            plans[current_group] = (new_group, updates)

        if not plans:
            return

        # Resolve the on-disk path; only `.name` is used, so open mode is moot
        with archive.m_context.raw_file(self.auxiliary_file) as file_handle:
            h5_path = file_handle.name

        applied_updates: list[tuple[ParameterSpaceSlice, str, str]] = []
        with h5py.File(h5_path, 'r+') as h5:
            # Two-phase move: park every source under a unique temporary name,
            # then move each temporary to its destination. A direct old->new
            # move would skip any rename whose destination is still occupied by
            # another group, silently corrupting permutation/cyclic renames
            # (e.g. /a_vs_b and /a_vs_c swapping).
            parked: list[tuple[str, str, list[tuple[ParameterSpaceSlice, str, str]]]]
            parked = []
            for index, (current_group, (new_group, updates)) in enumerate(
                plans.items()
            ):
                old_key = current_group.lstrip('/')
                if old_key not in h5:
                    logger.warning(
                        'Slice group missing from the auxiliary file; leaving '
                        'its references unchanged.',
                        group=current_group,
                        file=self.auxiliary_file,
                    )
                    continue
                temporary_key = f'__rename_tmp_{index}__'
                h5.move(old_key, temporary_key)
                parked.append((temporary_key, new_group.lstrip('/'), updates))
            for temporary_key, new_key, updates in parked:
                h5.move(temporary_key, new_key)
                applied_updates.extend(updates)

        # Only rewrite references for groups whose data was actually moved
        for parameter_slice, dataset, new_reference in applied_updates:
            setattr(parameter_slice, dataset, new_reference)

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
        from boss.pp.mesh import Mesh

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
        # Start from a clean file: a recomputation over an existing .h5 would
        # otherwise leave stale groups behind (e.g. slices named under a
        # previous set of parameter names)
        if archive.m_context.raw_path_exists(h5_filename):
            with archive.m_context.raw_file(h5_filename, 'rb') as existing:
                existing_path = existing.name
            os.remove(existing_path)
        handler = HDF5Handler(filename=h5_filename, archive=archive, logger=logger)

        iteration_procedure = np.arange(iter_no, 0, -1)

        group_names = slice_group_names(self.parameter_names)
        for parameter_counter, rank in enumerate(generate_slices(len(bounds))):
            main_rank, upper_rank = rank
            group = group_names[parameter_counter]
            fit_slices, uncertainty_slices = [], []

            # BOSS' Mesh builds the 2D slice grid over the two active dimensions
            # and fixes the rest to the global minimum (the 'min' preset resolves
            # to select('x_glmin')), replacing the hand-rolled query grid.
            mesh = Mesh(
                res.bounds,
                active_dims=[main_rank, upper_rank],
                grid_pts=no_grid_points,
            )
            for iteration in iteration_procedure:
                mesh.fix_dim_preset(res, 'min', itr=int(iteration))
                model = res.reconstruct_model(int(iteration))
                # one predict pass; evaluate_func splits the (mean, variance) tuple
                mean, variance = mesh.evaluate_func(lambda X, m=model: m.predict(X))
                fit_slices.append(np.asarray(mean))
                uncertainty_slices.append(np.sqrt(np.asarray(variance)))

            self.parameter_slices.append(ParameterSpaceSlice())

            archive_prefix = f'data.parameter_slices[{parameter_counter}]'
            for dataset, data in (
                ('fit', np.array(fit_slices)),
                ('uncertainty', np.array(uncertainty_slices)),
                ('iteration', iteration_procedure),
                ('parameters_x', np.array(compute_parameters(main_rank))),
                ('parameters_y', np.array(compute_parameters(upper_rank))),
            ):
                handler.add_dataset(
                    path=f'/{group}/{dataset}',
                    dataset=Dataset(
                        data=data,
                        archive_path=f'{archive_prefix}.{dataset}',
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
