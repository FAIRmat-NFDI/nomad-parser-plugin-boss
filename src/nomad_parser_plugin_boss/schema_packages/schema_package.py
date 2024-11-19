from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

import numpy as np

from nomad.config import config
from nomad.datamodel.data import Schema
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.datamodel.metainfo.plot import PlotSection, PlotlyFigure
from nomad.metainfo import Quantity, SchemaPackage

configuration = config.get_plugin_entry_point(
    'nomad_parser_plugin_boss.schema_packages:schema_package_entry_point'
)

m_package = SchemaPackage()


class PotentialEnergySurfaceFit(PlotSection, Schema):
    # m_def = Section()

    parameter_1_name = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )

    parameter_1_values = Quantity(
        type=np.float64,
        shape=['*'],
    )  # ! TODO use `PhysicalProperty`
    
    parameter_2_name = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )

    parameter_2_values = Quantity(
        type=np.float64,
        shape=['*'],
    )  # ! TODO use `PhysicalProperty`

    energy_values = Quantity(
        type=np.float64,
        unit='eV',  # ?
        shape=['*', '*'],
    )  # ! TODO use `PhysicalProperty`

    energy_variance = Quantity(
        type=np.float64,
        unit='eV^2',  # ?
        shape=['*', '*'],
    )  # ! TODO use `PhysicalProperty`

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super(Schema, self).normalize(archive, logger)

        try:
            if self.energy_values.shape != (len(self.parameter_1_values), len(self.parameter_2_values)):
                raise ValueError('Energy values shape does not match parameter values')  # ?

            self.figures.append(
                PlotlyFigure(
                    name='Potential Energy Surface',
                    data=[
                        dict(
                            x=self.parameter_values[0],
                            y=self.parameter_values[1],
                            z=self.energy_values,
                            type='contour',
                            labels=dict(
                                x=self.parameter_1_name,
                                y=self.parameter_2_name,
                                z='Energy',
                            ),
                        ),
                    ],
                )
            )
        except Exception as e:
            pass


m_package.__init_metainfo__()
