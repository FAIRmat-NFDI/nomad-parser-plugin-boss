from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

import numpy as np

from nomad.datamodel.data import Schema
from nomad.datamodel.hdf5 import HDF5Dataset
from nomad.datamodel.metainfo.annotations import (
    H5WebAnnotation,
    ELNAnnotation,
    ELNComponentEnum,
)
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection

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
            signal='./fitted_values',
            axes=['parameter_1_values', 'parameter_2_values'],
        )
    )

    fitted_values = Quantity(
        type=HDF5Dataset,
        unit='eV',
        a_h5web=H5WebAnnotation(long_name='PES', errors='fitted_stddevs'),
    )  # ? units

    fitted_stddevs = Quantity(type=HDF5Dataset, unit='eV')

    parameter_1_values = Quantity(
        type=HDF5Dataset, a_h5web=H5WebAnnotation(long_name='', indices=1)
    )

    parameter_2_values = Quantity(
        type=HDF5Dataset, a_h5web=H5WebAnnotation(long_name='', indices=2)
    )


class PotentialEnergySurfaceFit(Schema):
    m_def = Section(
        a_h5web=H5WebAnnotation(
            title='Potential Energy Surface Fit', paths=['parameter_slices/0']
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
                    parameter_slice.parameter_1_values.m_annotations[
                        'h5web'
                    ].long_name = self.parameter_names[main_rank]
                    parameter_slice.parameter_2_values.m_annotations[
                        'h5web'
                    ].long_name = self.parameter_names[upper_rank]
            else:
                logger.warning(
                    'Length mismatch between parameter names and slices. Not updating annotations.',
                    n_names=len(self.parameter_names),
                    n_slices=n_slices,
                )


m_package.__init_metainfo__()
