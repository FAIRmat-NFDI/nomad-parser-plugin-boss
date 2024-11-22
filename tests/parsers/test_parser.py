import logging

from nomad.datamodel import EntryArchive

from nomad_parser_plugin_boss.parsers.parser import BossPostProcessingParser


def test_parse_file():
    parser = BossPostProcessingParser()
    archive = EntryArchive()
    parser.parse('tests/data/example.out', archive, logging.getLogger())

    assert archive.workflow2.name == 'test'
