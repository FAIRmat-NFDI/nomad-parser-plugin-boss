import re
from collections import Counter
from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from nomad.datamodel.data import ArchiveSection, EntryData, Schema
from nomad.datamodel.hdf5 import HDF5Reference
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    ELNComponentEnum,
    H5WebAnnotation,
    SectionProperties,
)
from nomad.datamodel.metainfo.plot import PlotlyFigure
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection
from nomad_bayesian_optimization.naming import sanitize_quantity_name
from nomad_bayesian_optimization.schema_packages.bayesian_optimization import (
    BayesianOptimization,
    ContinuousParameter,
    Objective,
    Target,
)
from nomad_bayesian_optimization.step_schema import (
    FieldSpec,
    attach_step_package,
    make_step_instance,
)
from nomad_measurements.utils import Dataset, HDF5Handler
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots

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

    name = Quantity(
        type=str,
        description="""
        Label of the compared-parameter slice, e.g. `alpha_vs_beta`, matching the
        H5Web group name. Used as the subsection label so the archive tree shows
        the parameter combination instead of the list index.
        """,
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


class Acquisitions(ArchiveSection):
    """BOSS acquisition history, mirroring BOSS' ``plot_acquisitions``: the sampled
    points and the predicted global minimum tracked per iteration."""

    # Predicted global-minimum series (one entry per BO iteration).
    iteration = Quantity(type=int, shape=['*'])
    predicted_minimum = Quantity(
        type=np.float64,
        unit='eV',
        shape=['*'],
        description='Predicted global-minimum energy per iteration.',
    )
    predicted_minimum_uncertainty = Quantity(
        type=np.float64,
        unit='eV',
        shape=['*'],
        description="""
        Std. dev. of the predicted global minimum (sqrt of its GP variance).
        """,
    )
    predicted_minimum_location = Quantity(
        type=np.float64,
        shape=['*', '*'],
        description="""
        Predicted global-minimum location per iteration
        (row: iteration, column: parameter).
        """,
    )

    # Acquired points (one entry per acquisition; the initial batch shares
    # iteration 0, so ``acquisition_iteration`` may repeat).
    acquisition_iteration = Quantity(type=int, shape=['*'])
    acquired_value = Quantity(
        type=np.float64,
        unit='eV',
        shape=['*'],
        description='Objective value observed at each acquired point.',
    )
    acquired_location = Quantity(
        type=np.float64,
        shape=['*', '*'],
        description='Acquired point location (row: acquisition, column: parameter).',
    )


def extract_acquisitions(res, logger: 'BoundLogger') -> 'Acquisitions | None':
    """Read BOSS' acquisition history straight off ``BOResults``.

    This is the exact data behind BOSS' ``plot_acquisitions``, with no model
    re-fit needed: the sampled points ``(X, Y)`` and the predicted global minimum
    (value, uncertainty and location) tracked per iteration. Returns ``None`` if
    the results cannot be read.
    """
    try:
        return Acquisitions(
            iteration=np.asarray(list(res['x_glmin'].keys()), dtype=int),
            predicted_minimum=res['mu_glmin'].to_array(),
            predicted_minimum_uncertainty=np.sqrt(
                np.clip(res['nu_glmin'].to_array(), 0.0, None)
            ),
            predicted_minimum_location=res['x_glmin'].to_array(),
            acquisition_iteration=np.asarray(
                res.batch_tracker.iteration_labels, dtype=int
            ),
            acquired_value=np.asarray(res['Y'])[:, 0],
            acquired_location=np.asarray(res['X']),
        )
    except Exception as e:
        logger.warning('Could not extract BOSS acquisition history.', error=str(e))
        return None


def build_campaign_steps(archive, parameter_names: list[str], locations, values):
    """Record each acquired point as a measured ``Step``.

    Reuses the nomad-bayesian-optimization step machinery (no BayBE needed): a
    per-campaign ``CampaignStep`` subclass with one numeric column per parameter
    plus the ``energy`` target, each tagged with ``baybe_name`` so the inherited
    counters, per-step table and progress figure populate. Every acquisition is a
    recorded measurement (``recommended=False``), never a pending recommendation.
    """
    field_specs = [
        FieldSpec(
            name=name, quantity_name=sanitize_quantity_name(name), type='float'
        )
        for name in parameter_names
    ]
    field_specs.append(
        FieldSpec(
            name='energy',
            quantity_name=sanitize_quantity_name('energy'),
            type='float',
            unit='eV',
            is_target=True,
        )
    )
    step_def = attach_step_package(archive, field_specs)
    steps = []
    for location, value in zip(locations, values):
        record = {name: float(location[i]) for i, name in enumerate(parameter_names)}
        record['energy'] = float(value)
        steps.append(
            make_step_instance(step_def, record, field_specs, recommended=False)
        )
    return steps


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

    # `label_quantity='name'` tells the GUI to label each repeating slice by its
    # `name` (e.g. `alpha_vs_beta`) instead of the list index.
    parameter_slices = SubSection(
        sub_section=ParameterSpaceSlice.m_def, repeats=True, label_quantity='name'
    )

    acquisitions = SubSection(sub_section=Acquisitions.m_def)

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

        group_names = slice_group_names(self.parameter_names)
        self._rename_slice_groups(archive, group_names, logger)
        # Keep the subsection label in step with the (possibly renamed) groups so
        # the archive tree shows `alpha_vs_beta` rather than the list index. Also
        # backfills the name on entries computed before this field existed.
        for parameter_slice, group_name in zip(self.parameter_slices, group_names):
            parameter_slice.name = group_name

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

    def _acquisitions_figure(self):
        """BOSS acquisition history as a two-panel Plotly figure, in the same style
        as the nomad-bayesian-optimization progress plots (a PlotlyFigure on
        ``figures``). Mirrors BOSS' ``plot_acquisitions``:

        * top: acquired objective values with the predicted global minimum
          (``mu`` and its ``mu +/- sigma`` band);
        * bottom: acquired locations with the predicted global-minimum location,
          one trace pair per parameter (labelled by parameter name).
        """
        acq = self.acquisitions
        if (
            acq is None
            or acq.acquired_value is None
            or len(acq.acquired_value) == 0
        ):
            return None

        def _floats(values):
            return [float(getattr(v, 'magnitude', v)) for v in values]

        acq_iters = [int(i) for i in acq.acquisition_iteration]
        acquired = _floats(acq.acquired_value)
        acquired_loc = np.asarray(acq.acquired_location, dtype=float)

        min_iters = [int(i) for i in acq.iteration]
        mu = _floats(acq.predicted_minimum)
        std = (
            _floats(acq.predicted_minimum_uncertainty)
            if acq.predicted_minimum_uncertainty is not None
            else [0.0] * len(mu)
        )
        min_loc = np.asarray(acq.predicted_minimum_location, dtype=float)
        names = list(self.parameter_names or [])

        figure = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.18
        )

        # Top panel: predicted minimum (mu +/- sigma) and acquired values.
        upper = [m + s for m, s in zip(mu, std)]
        lower = [m - s for m, s in zip(mu, std)]
        figure.add_trace(
            go.Scatter(
                x=min_iters + min_iters[::-1],
                y=upper + lower[::-1],
                fill='toself',
                fillcolor='rgba(0,0,0,0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                hoverinfo='skip',
                showlegend=False,
                name='Uncertainty',
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=min_iters, y=mu, mode='lines+markers', name='Predicted minimum μ'
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=acq_iters,
                y=acquired,
                mode='markers',
                marker=dict(symbol='circle-open'),
                name='Acquired value',
            ),
            row=1,
            col=1,
        )

        # Bottom panel: acquired and predicted-minimum locations, per parameter.
        n_dims = acquired_loc.shape[1] if acquired_loc.ndim > 1 else 0
        for i in range(n_dims):
            label = names[i] if i < len(names) else f'parameter_{i}'
            color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
            figure.add_trace(
                go.Scatter(
                    x=acq_iters,
                    y=list(acquired_loc[:, i]),
                    mode='markers',
                    marker=dict(symbol='circle-open', color=color),
                    name=f'{label} (acquired)',
                    legendgroup=label,
                ),
                row=2,
                col=1,
            )
            if min_loc.ndim > 1 and i < min_loc.shape[1]:
                figure.add_trace(
                    go.Scatter(
                        x=min_iters,
                        y=list(min_loc[:, i]),
                        mode='lines',
                        line=dict(color=color),
                        name=f'{label} (predicted min)',
                        legendgroup=label,
                    ),
                    row=2,
                    col=1,
                )

        figure.update_xaxes(tickformat='d', title_text='Iteration', row=2, col=1)
        figure.update_yaxes(title_text='y and predicted minimum μ (eV)', row=1, col=1)
        figure.update_yaxes(title_text='x and predicted-min location', row=2, col=1)
        # One legend per panel, each sitting just above its own subplot. With
        # vertical_spacing=0.18 row 1 spans the paper domain [0.59, 1.0] and row 2
        # [0.0, 0.41], so legend2 sits in the gap just above row 2.
        figure.update_traces(legend='legend', row=1, col=1)
        figure.update_traces(legend='legend2', row=2, col=1)
        figure.update_layout(
            template='plotly_white',
            height=800,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
            legend2=dict(orientation='h', yanchor='bottom', y=0.44, x=0),
        )
        return PlotlyFigure(label='acquisitions', figure=figure.to_plotly_json())

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        super().normalize(archive, logger)
        self.refresh_h5web_labels(archive, logger)


class ELNBOSSAnalysis(PotentialEnergySurfaceFit, BayesianOptimization):
    """
    ELN entry for BOSS Bayesian Optimization analysis results.

    Inherits the generic `BayesianOptimization` schema (parameters, objective,
    steps, status, counters, progress figures) so a BOSS run surfaces as a
    first-class Bayesian-optimization entry, and extends it with the BOSS-specific
    potential-energy-surface slices (`parameter_slices` + H5Web). `BayesianOptimization`
    already provides `PlotSection` + `EntryData`.
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

        # Append the acquisitions figure after BayesianOptimization.normalize rebuilds
        # self.figures from the (currently empty) steps, so it is not clobbered.
        figure = self._acquisitions_figure()
        if figure is not None:
            self.figures = list(self.figures or []) + [figure]

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

        # Populate the inherited BayesianOptimization schema: continuous parameters
        # (from the search-space bounds) and a single energy objective to minimise.
        # The PES slices are added below; per-step records are left for a follow-up.
        self.search_space_type = 'Continuous'
        self.status = 'Finished'
        self.parameters = [
            ContinuousParameter(
                name=self.parameter_names[rank],
                lower_bound=float(bounds[rank][0]),
                upper_bound=float(bounds[rank][1]),
            )
            for rank in range(len(bounds))
        ]
        self.objective = Objective(
            type='SingleTargetObjective',
            targets=[Target(name='energy', type='NumericalTarget', mode='MIN')],
        )

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

        # Acquisition history (sampled points + predicted global minimum per
        # iteration); drives the two-panel Acquisitions figure.
        self.acquisitions = extract_acquisitions(res, logger)
        # Record each acquired point as a measured Step so the inherited
        # BayesianOptimization counters, step table and progress figure populate.
        if self.acquisitions is not None:
            self.steps = build_campaign_steps(
                archive,
                list(self.parameter_names),
                np.asarray(res['X']),
                np.asarray(res['Y'])[:, 0],
            )

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

            self.parameter_slices.append(ParameterSpaceSlice(name=group))

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
