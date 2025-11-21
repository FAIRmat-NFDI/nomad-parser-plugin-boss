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


class ParameterSpaceSlice(ArchiveSection):
    # ! TODO use `PhysicalProperty`
    m_def = Section(
        label='Parameter Space Slice',
        a_h5web=H5WebAnnotation(
            signal='fit',
            auxiliary_signals=['uncertainty'],
            axes=['iteration', 'parameters_x', 'parameters_y'],
        )
    )

    fit = Quantity(
        type=HDF5Reference,
        unit='eV',
        a_h5web=H5WebAnnotation(long_name='Potential Energy Surface Fit'),
    )

    uncertainty = Quantity(
        type=HDF5Reference,
        unit='eV',
    )

    iteration = Quantity(
        type=HDF5Reference,
        description="""
        Iteration number of the data aggregation process in the fit.
        The starting data is labeled as 0
        """,
    )

    parameters_x = Quantity(
        type=HDF5Reference,
        a_h5web=H5WebAnnotation(long_name='a'),
    )

    parameters_y = Quantity(
        type=HDF5Reference,
        a_h5web=H5WebAnnotation(long_name='b'),
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

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        if isinstance(self.parameter_names, list):
            if len(self.parameter_names) == (n_slices := len(self.parameter_slices)):
                for slice_indices, parameter_slice in zip(
                    generate_slices(n_slices), self.parameter_slices
                ):
                    main_rank, upper_rank = slice_indices
                    parameter_slice.parameters_x.m_annotations[
                        'h5web'
                    ].long_name = self.parameter_names[main_rank]
                    parameter_slice.parameters_y.m_annotations[
                        'h5web'
                    ].long_name = self.parameter_names[upper_rank]
            else:
                logger.warning(
                    (
                        'Length mismatch between parameter names and slices. ',
                        'Not updating annotations.'
                    ),
                    n_names=len(self.parameter_names),
                    n_slices=n_slices,
                )


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
        Normalization method for ELN entry.
        Parses BOSS data and populates HDF5 auxiliary file with PES data.
        """
        import os

        from boss.bo.results import BOResults
        from boss.io.dump import build_query_points
        from boss.pp.pp_main import PPMain

        super().normalize(archive, logger)

        # Auto-set entry name from file if not set
        if self.data_file and not archive.metadata.entry_name:
            file_base = os.path.basename(self.data_file)
            archive.metadata.entry_name = f'BOSS Analysis: {file_base}'

        # Parse BOSS data if not already done
        if self.data_file and not self.parameter_slices:
            logger.info('Parsing BOSS data in normalize()', data_file=self.data_file)

            try:
                # Get the mainfile path
                mainfile = os.path.join(archive.m_context.raw_path(), self.data_file)
                boss_out = os.path.join(os.path.dirname(mainfile), 'boss.out')

                # Load BOSS results
                res = BOResults.from_file(mainfile, boss_out)

                iter_no = res.settings.get('iterpts', 1)
                no_grid_points = 50  # Can be made configurable

                # Get bounds for parameter space
                bounds = res.settings.get('bounds', [])

                if bounds is None or len(bounds) == 0:
                    logger.warning('No bounds found in BOSS results')
                    return

                # Helper function to compute parameter values
                def compute_parameters(rank: int):
                    return np.linspace(
                        bounds[rank][0], bounds[rank][1], num=no_grid_points
                    )

                # Store parameter names if available
                self.parameter_names = [f'parameter_{i}' for i in range(len(bounds))]

                # Create HDF5 handler for auxiliary file
                h5_filename = f'{self.data_file.rsplit(".", 1)[0]}.h5'
                self.auxiliary_file = h5_filename
                handler = HDF5Handler(
                    filename=h5_filename, archive=archive, logger=logger
                )

                # Generate slices for all parameter combinations
                iteration_procedure = np.arange(iter_no, 0, -1)
                slices_list = list(generate_slices(len(bounds)))

                for parameter_counter, rank in enumerate(slices_list):
                    main_rank, upper_rank = rank
                    mu_all_slices, var_all_slices = [], []

                    for iteration in iteration_procedure:
                        pp = PPMain(
                            res,
                            pp_models=True,
                            pp_iters=[iteration],
                            pp_model_slice=[
                                main_rank + 1,
                                upper_rank + 1,
                                no_grid_points,
                            ],
                        )

                        X = build_query_points(
                            pp.settings, res.select('x_glmin', iter_no)
                        )

                        mu, var = res.reconstruct_model(iteration).predict(X)
                        mu_all_slices.append(mu.reshape(no_grid_points, no_grid_points))
                        var_all_slices.append(var.reshape(no_grid_points, no_grid_points))

                    # Create the slice section (this creates the subsection in archive)
                    slice_path = f'parameter_slices/{parameter_counter}'
                    self.m_setdefault(slice_path)

                    # Add datasets to HDF5 handler with archive paths (use square brackets for array indices)
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
                            archive_path=f'data.parameter_slices[{parameter_counter}].uncertainty',
                        ),
                    )

                    handler.add_dataset(
                        path=f'/slice_{parameter_counter}/iteration',
                        dataset=Dataset(
                            data=iteration_procedure,
                            archive_path=f'data.parameter_slices[{parameter_counter}].iteration',
                        ),
                    )

                    handler.add_dataset(
                        path=f'/slice_{parameter_counter}/parameters_x',
                        dataset=Dataset(
                            data=np.array(compute_parameters(main_rank)),
                            archive_path=f'data.parameter_slices[{parameter_counter}].parameters_x',
                        ),
                    )

                    handler.add_dataset(
                        path=f'/slice_{parameter_counter}/parameters_y',
                        dataset=Dataset(
                            data=np.array(compute_parameters(upper_rank)),
                            archive_path=f'data.parameter_slices[{parameter_counter}].parameters_y',
                        ),
                    )

                    # Add NeXus metadata for H5Web visualization
                    handler.add_attribute(
                        path=f'/slice_{parameter_counter}',
                        params=dict(
                            axes=['iteration', 'parameters_x', 'parameters_y'],
                            signal='fit',
                            auxiliary=['uncertainty'],
                            NX_class='NXdata',
                        ),
                    )

                # Write HDF5 file and populate HDF5Reference quantities
                handler.write_file()

                # Initialize figures to trigger Overview tab visualization
                # Even though we use H5Web (not Plotly), PlotSection may require this
                self.figures = []

                logger.info(
                    'Successfully parsed BOSS data and created HDF5 file',
                    n_slices=len(self.parameter_slices),
                    n_iterations=len(iteration_procedure),
                    h5_file=h5_filename,
                )

            except Exception as e:
                logger.error('Error parsing BOSS data', error=str(e), exc_info=True)
                raise


class RawFileBOSSData(EntryData):
    """
    Entry section for a BOSS raw data file.
    Contains only a reference to the main ELN measurement entry.
    """

    measurement = Quantity(
        type=ELNBOSSAnalysis,
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        ),
    )


m_package.__init_metainfo__()
