from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from structlog.stdlib import (
        BoundLogger,
    )

import numpy as np

from nomad.config import config
from nomad.datamodel.datamodel import EntryArchive
from nomad.parsing.parser import MatchingParser
from nomad.parsing.file_parser.text_parser import TextParser, Quantity as TextQuantity

from nomad_parser_plugin_boss.schema_packages.schema_package import PotentialEnergySurfaceFit

configuration = config.get_plugin_entry_point(
    'nomad_parser_plugin_boss.parsers:parser_entry_point'
)


class BossSliceParser(TextParser):
    def init_quantities(self):
        def split_row(full_row: str) -> dict[str, float]:
            split_row = full_row.split()
            key_defs = ('x_1', 'x_2', 'mu', 'nu') if len(split_row) == 4 else ('x_1', 'mu', 'nu')
            return {k: float(x) for k, x in zip(key_defs, split_row)}

        # re_float = r'\d\.\d+e[\-\+]\d{2}'

        self._quantities = [
            TextQuantity(
                'row',
                r'((?:\s+\d\.\d+e[\-\+]\d{2}){3,4})\n',
                str_operation=split_row,
                repeats=True,
            ),
        ]


class BossPostProcessingParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        logger.info('BossPostProcessingParser.parse', parameter=configuration.parameter)

        slice_parser = BossSliceParser(mainfile=mainfile, logger=logger)
        slice_parser.parse()

        def get_column_unique(column_name: str) -> list[float]:
            return np.sort(list({x.get(column_name) for x in slice_parser.results.get('row', [])}))

        def get_column(column_name: str) -> list[float]:
            return [x.get(column_name) for x in slice_parser.results.get('row', [])]

        def reshaping(target: list, dim_1: int, dim_2: int) -> np.ndarray:
            if dim_2:
                return np.reshape(target, (dim_1, dim_2))
            else:
                return np.reshape(target, (dim_1, -1))

        x_1, x_2 = get_column_unique('x_1'), get_column_unique('x_2')

        archive.m_add_sub_section(EntryArchive.data, 
            PotentialEnergySurfaceFit(
                parameter_1_name='parameter_1_name',
                parameter_1_values=x_1,
                parameter_2_name='parameter_2_name',
                parameter_2_values=x_2,
                energy_values=reshaping(get_column('mu'), len(x_1), len(x_2)),
                energy_variance=reshaping(get_column('nu'), len(x_1), len(x_2)),
            )                      
        )
