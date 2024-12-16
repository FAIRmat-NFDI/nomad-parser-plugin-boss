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


class FittedValue(Schema):
    m_def = Section(
        a_h5web=H5WebAnnotation(
            signal='signal',
            axes=['parameter_2', 'parameter_1'],
        )
    )

    signal = Quantity(
        type=HDF5Dataset,
        unit='eV',  # ?
        shape=[],
    )  # ! TODO use `PhysicalProperty`

    parameter_names = Quantity(
        type=str,
        shape=['*'],
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )

    parameter_1 = Quantity(
        type=HDF5Dataset,
        shape=[],
    )  # ! TODO use `PhysicalProperty`

    parameter_2 = Quantity(
        type=HDF5Dataset,
        shape=[],
    )  # ! TODO use `PhysicalProperty`


class PotentialEnergySurfaceFit(Schema):
    m_def = Section(
        a_h5web=H5WebAnnotation(
            title='Potential Energy Surface Fit',
            paths=['energy_values', 'energy_std'],
        ),
    )

    energy_values = SubSection(sub_section=FittedValue.m_def)

    energy_std = SubSection(sub_section=FittedValue.m_def)


m_package.__init_metainfo__()
