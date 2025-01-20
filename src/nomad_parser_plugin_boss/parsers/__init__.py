from nomad.config.models.plugins import ParserEntryPoint
from pydantic import Field


class BossParserEntryPoint(ParserEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_parser_plugin_boss.parsers.parser import BossPostProcessingParser

        return BossPostProcessingParser(**self.dict())


parser_entry_point = BossParserEntryPoint(
    name='BossParser',
    description='New parser entry point configuration.',
    mainfile_name_re='.*\.rst',
)
