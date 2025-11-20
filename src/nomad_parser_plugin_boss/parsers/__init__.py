from nomad.config.models.plugins import ParserEntryPoint


class BossParserEntryPoint(ParserEntryPoint):
    def load(self):
        from nomad_parser_plugin_boss.parsers.parser import BossParser

        return BossParser(**self.dict())


parser_entry_point = BossParserEntryPoint(
    name='BossParser',
    description='Parser for BOSS Bayesian Optimization result files (.rst format).',
    mainfile_name_re=r'^.*\.rst$',
    mainfile_mime_re='text/.*|application/octet-stream',
)
