from collections.abc import Generator
from typing import TYPE_CHECKING

from nomad.datamodel.data import EntryData, Schema
from nomad.datamodel.hdf5 import HDF5Dataset
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    ELNComponentEnum,
    H5WebAnnotation,
    SectionProperties,
)
from nomad.datamodel.metainfo.plot import PlotSection
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger


m_package = SchemaPackage()


def generate_slices(ranks: int) -> Generator:
    """Produce all possible index pairs defining slices of the parameter space."""
    for main_rank in range(ranks):
        for upper_rank in range(main_rank + 1, ranks):
            yield main_rank, upper_rank


class ParameterSpaceSlice(Schema):
    # ! TODO use `PhysicalProperty`
    m_def = Section(
        a_h5web=H5WebAnnotation(
            signal='fit',
            errors='uncertainty',  # only relevant for 1-D visualizations
            axes=['iteration', 'parameters_x', 'parameters_y'],
        )
    )

    fit = Quantity(
        type=HDF5Dataset,
        unit='eV',
        a_h5web=H5WebAnnotation(long_name='Potential Energy Surface Fit'),
    )  # ? units

    uncertainty = Quantity(type=HDF5Dataset, unit='eV')

    iteration = Quantity(
        type=HDF5Dataset,
        description="""
        Iteration number of the data aggregation process in the fit.
        The starting data is labeled as 0
        """,
    )

    parameters_x = Quantity(type=HDF5Dataset, a_h5web=H5WebAnnotation(long_name='a'))

    parameters_y = Quantity(type=HDF5Dataset, a_h5web=H5WebAnnotation(long_name='b'))


class PotentialEnergySurfaceFit(Schema):
    m_def = Section(
        a_h5web=H5WebAnnotation(paths=['parameter_slices/0']),
    )

    data_file = Quantity(
        type=str,
        description='Path to the BOSS .rst data file',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.FileEditQuantity,
        ),
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
        """
        super().normalize(archive, logger)

        # Auto-set entry name from file if not set
        if self.data_file and not archive.metadata.entry_name:
            import os

            file_base = os.path.basename(self.data_file)
            archive.metadata.entry_name = f'BOSS Analysis: {file_base}'


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
