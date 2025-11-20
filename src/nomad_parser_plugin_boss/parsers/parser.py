import os
from typing import TYPE_CHECKING

import numpy as np
from boss.bo.results import BOResults
from boss.io.dump import build_query_points
from boss.pp.pp_main import PPMain
from nomad.config import config
from nomad.datamodel.context import ServerContext
from nomad.parsing.parser import MatchingParser
from nomad_measurements.utils import create_archive

from nomad_parser_plugin_boss.schema_packages.schema_package import (
    ELNBOSSAnalysis,
    RawFileBOSSData,
    generate_slices,
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
        1. Creates an ELN measurement entry with all the data
        2. Creates a reference to that entry
        3. Sets the current archive as a RawFile entry pointing to the measurement
        """
        logger.info('BossParser.parse', mainfile=mainfile)

        # Get the data file name (handle both local and server contexts)
        data_file = mainfile.split('/')[-1]
        if isinstance(archive.m_context, ServerContext):
            data_file = mainfile.split('/raw/', 1)[1]

        # Create the ELN measurement entry
        entry = ELNBOSSAnalysis()
        entry.data_file = data_file

        # Parse the BOSS data and populate the entry
        self._parse_boss_data(mainfile, entry, logger)

        # Create the archive file for the ELN entry
        # This will be named like: "my_boss_file.archive.json"
        file_name = f'{"".join(data_file.split(".")[:-1])}.archive.json'

        # Set this archive as a RawFile entry with reference to the measurement
        archive.data = RawFileBOSSData(
            measurement=create_archive(entry, archive, file_name)
        )
        archive.metadata.entry_name = f'{data_file} data file'

    def _parse_boss_data(
        self,
        mainfile: str,
        entry: ELNBOSSAnalysis,
        logger: 'BoundLogger',
    ) -> None:
        """
        Parse BOSS .rst file and populate the ELN entry with data.

        This is the core parsing logic extracted from the original parser.
        """
        try:
            # Load BOSS results
            res = BOResults.from_file(
                mainfile, os.path.join(os.path.dirname(mainfile), 'boss.out')
            )

            iter_no = res.settings.get('iterpts', 1)
            no_grid_points = 50  # Can be made configurable

            # Get bounds for parameter space
            bounds = res.settings.get('bounds', [])

            if bounds is None or len(bounds) == 0:
                logger.warning('No bounds found in BOSS results')
                return

            # Helper function to compute parameter values
            def compute_parameters(rank: int):
                return np.linspace(bounds[rank][0], bounds[rank][1], num=no_grid_points)

            # Store parameter names if available
            # (You can extract these from BOSS settings if available)
            entry.parameter_names = [f'parameter_{i}' for i in range(len(bounds))]

            # Generate slices for all parameter combinations
            iteration_procedure = np.arange(iter_no, 0, -1)

            for parameter_counter, rank in enumerate(generate_slices(len(bounds))):
                main_rank, upper_rank = rank
                mu_all_slices, var_all_slices = [], []

                for iteration in iteration_procedure:
                    pp = PPMain(
                        res,
                        pp_models=True,
                        pp_iters=[iteration],
                        pp_model_slice=[main_rank + 1, upper_rank + 1, no_grid_points],
                    )

                    X = build_query_points(pp.settings, res.select('x_glmin', iter_no))

                    mu, var = res.reconstruct_model(iteration).predict(X)
                    mu_all_slices.append(mu.reshape(no_grid_points, no_grid_points))
                    var_all_slices.append(var.reshape(no_grid_points, no_grid_points))

                # Save slices to the entry
                slice_path = f'parameter_slices/{parameter_counter}'
                section = entry.m_setdefault(slice_path)

                section.fit = np.array(mu_all_slices)
                section.uncertainty = np.sqrt(np.array(var_all_slices))
                section.iteration = iteration_procedure
                section.parameters_x = np.array(compute_parameters(main_rank))
                section.parameters_y = np.array(compute_parameters(upper_rank))

            logger.info(
                'Successfully parsed BOSS data',
                n_slices=len(entry.parameter_slices),
                n_iterations=len(iteration_procedure),
            )

        except Exception as e:
            logger.error('Error parsing BOSS data', error=str(e), exc_info=True)
            raise
