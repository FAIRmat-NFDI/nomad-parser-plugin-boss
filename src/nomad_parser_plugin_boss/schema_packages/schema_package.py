from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

import numpy as np

from nomad.datamodel.data import Schema
from nomad.datamodel.hdf5 import HDF5Dataset
from nomad.datamodel.metainfo.annotations import (
    H5WebAnnotation,
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
            signal='fit',
            axes=['parameters_x', 'parameters_y'],
        )
    )

    fit = Quantity(
        type=HDF5Dataset,
        unit='eV',
        a_h5web=H5WebAnnotation(
            long_name='Potential Energy Surface Fit',
            errors='fitting_errors',
        ),
    )  # ? units

    fitting_errors = Quantity(type=HDF5Dataset, unit='eV')

    parameters_x = Quantity(
        type=HDF5Dataset, a_h5web=H5WebAnnotation(long_name='a', indices=1)
    )

    parameters_y = Quantity(
        type=HDF5Dataset, a_h5web=H5WebAnnotation(long_name='b', indices=2)
    )


class PotentialEnergySurfaceFit(Schema):
    m_def = Section(
        a_h5web=H5WebAnnotation(paths=['parameter_slices/0']),
    )

    n_parameters = Quantity(type=int)

    parameter_names = Quantity(
        type=str,
        shape=['*'],
        a_eln=dict(component='StringEditQuantity'),
    )

    parameter_slices = SubSection(sub_section=ParameterSpaceSlice.m_def, repeats=True)

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        if isinstance(self.parameter_names, list):
            if len(self.parameter_names) == self.n_parameters:
                for slice_indices, parameter_slice in zip(
                    generate_slices(self.n_parameters), self.parameter_slices
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
                    'Length mismatch between parameter names and slices. Not updating annotations.',
                    n_names=len(self.parameter_names),
                    n_parameters=self.n_parameters,
                )


m_package.__init_metainfo__()
