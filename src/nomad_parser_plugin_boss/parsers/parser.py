from typing import TYPE_CHECKING

from nomad.config import config
from nomad.datamodel.context import ServerContext
from nomad.parsing.parser import MatchingParser
from nomad_measurements.utils import create_archive

from nomad_parser_plugin_boss.schema_packages.schema_package import (
    ELNBOSSAnalysis,
    RawFileBOSSData,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

configuration = config.get_plugin_entry_point(
    'nomad_parser_plugin_boss.parsers:parser_entry_point'
)


class BossParser(MatchingParser):
    """
    Parser for BOSS .rst files that creates dual entries:
    1. RawFileBOSSData entry (for the data file itself)
    2. ELNBOSSAnalysis entry (for the measurement/analysis)

    The actual BOSS data parsing happens in ELNBOSSAnalysis.normalize()
    which is called during entry processing after create_archive().
    """

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        """
        Parse BOSS .rst file and create dual-entry architecture.

        This method:
        1. Creates an ELN measurement entry with data_file set
        2. Calls create_archive() which triggers normalize() for BOSS data parsing
        3. Sets the current archive as a RawFile entry pointing to the measurement
        """
        logger.debug('BossParser.parse', mainfile=mainfile)

        # Get the data file name (handle both local and server contexts)
        data_file = mainfile.split('/')[-1]
        if isinstance(archive.m_context, ServerContext):
            data_file = mainfile.split('/raw/', 1)[1]

        # Create the ELN measurement entry with data_file
        entry = ELNBOSSAnalysis()
        entry.data_file = data_file
        entry.m_context = archive.m_context

        # Create the archive file for the ELN entry
        # This will be named like: "my_boss_file.archive.json"
        # The normalize() method will be called by the archive processing system
        file_name = f'{"".join(data_file.split(".")[:-1])}.archive.json'

        # Set this archive as a RawFile entry with reference to the measurement
        archive.data = RawFileBOSSData(
            measurement=create_archive(entry, archive, file_name)
        )
        archive.metadata.entry_name = f'{data_file} data file'
