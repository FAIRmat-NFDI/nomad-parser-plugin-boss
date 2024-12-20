import numpy as np

from nomad.datamodel.data import Schema, ArchiveSection
from nomad.datamodel.hdf5 import HDF5Dataset
from nomad.datamodel.metainfo.annotations import (
    H5WebAnnotation,
    ELNAnnotation,
    ELNComponentEnum,
)
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection

m_package = SchemaPackage()


class ParameterSpaceSlice(Schema):
    # ! TODO use `PhysicalProperty`
    m_def = Section(
        a_h5web=H5WebAnnotation(
            signal='fitted_values',
            axes=['blank', 'parameter_2_values', 'parameter_1_values'],
        )
    )

    fitted_values = Quantity(
        type=HDF5Dataset, unit='eV', a_h5web=H5WebAnnotation(long_name='PES')
    )  # ! TODO: add errors # ? units

    parameter_1_values = Quantity(
        type=HDF5Dataset, a_h5web=H5WebAnnotation(long_name='')
    )

    parameter_2_values = Quantity(
        type=HDF5Dataset, a_h5web=H5WebAnnotation(long_name='')
    )

    blank = Quantity(
        type=HDF5Dataset, a_h5web=H5WebAnnotation()
    )

    def normalize(self, archive, logger):
        self.blank = np.array([])


class PotentialEnergySurfaceFit(Schema):
    m_def = Section(
        a_h5web=H5WebAnnotation(title='Potential Energy Surface Fit', paths=['parameter_slices/0']),
    )

    parameter_names = Quantity(
        type=str,
        shape=['*'],
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )

    parameter_slices = SubSection(sub_section=ParameterSpaceSlice.m_def, repeats=True)


m_package.__init_metainfo__()
