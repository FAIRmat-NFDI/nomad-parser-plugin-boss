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

from nomad.config import config
from nomad.parsing.parser import MatchingParser
from nomad.parsing.file_parser.text_parser import TextParser, Quantity as TextQuantity

from nomad_parser_plugin_boss.schema_packages.schema_package import PotentialEnergySurfaceFit

configuration = config.get_plugin_entry_point(
    'nomad_parser_plugin_boss.parsers:parser_entry_point'
)


class BossSliceParser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):
        def split_row(full_row: str, key_defs: tuple[str] = ('x_1', 'x_2', 'mu', 'nu')) -> dict[str, float]:
            return {k: float(x) for k, x in zip(key_defs, full_row.split())}

        # re_float = r'\d\.\d+e[\-\+]\d{2}'

        table_quantities = [
            TextQuantity(
                'row',
                r'((?:\s+\d\.\d+e[\-\+]\d{2}){4})\n',
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

        archive.data = [
            PotentialEnergySurfaceFit(
                parameter_1_name='parameter_1_name',
                parameter_1_values=set(slice_parser.get_quantity('row').data[0]['x_1']),
                parameter_2_name='parameter_2_name',
                parameter_2_values=set(slice_parser.get_quantity('row').data[0]['x_2']),
                energy_values=slice_parser.get_quantity('row').data[0]['mu'],
                energy_variance=slice_parser.get_quantity('row').data[0]['nu'],
            )
        ]
        
