import numpy as np

from nomad.datamodel.data import Schema
from nomad.datamodel.hdf5 import HDF5Dataset
from nomad.datamodel.metainfo.annotations import H5WebAnnotation, ELNAnnotation, ELNComponentEnum
from nomad.metainfo import Quantity, SchemaPackage, Section

m_package = SchemaPackage()


class PotentialEnergySurfaceFit(Schema):
    m_def = Section(
        a_h5web=H5WebAnnotation(
            title='Potential Energy Surface Fit',
            axes=['parameter_2', 'parameter_1'],
            signal='energy_values',
            # auxialary_singals=['energy_variance'],
        ),
    )

    parameter_names = Quantity(
        type=str,
        shape=['*'],
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )

    parameter_1 = Quantity(
        type=HDF5Dataset,
        # unit='',
        shape=[],
    )  # ! TODO use `PhysicalProperty`

    parameter_2 = Quantity(
        type=HDF5Dataset,
        # unit='',
        shape=[],
    )  # ! TODO use `PhysicalProperty`

    energy_values = Quantity(
        type=HDF5Dataset,
        unit='eV',  # ?
        shape=[],
    )  # ! TODO use `PhysicalProperty`

    energy_variance = Quantity(
        type=HDF5Dataset,
        unit='eV^2',  # ?
        shape=[],
    )  # ! TODO use `PhysicalProperty`


m_package.__init_metainfo__()
