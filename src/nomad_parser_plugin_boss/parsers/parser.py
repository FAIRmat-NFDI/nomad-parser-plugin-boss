import os
from typing import TYPE_CHECKING

import numpy as np
from boss.bo.results import BOResults
from boss.io.dump import build_query_points
from boss.pp.pp_main import PPMain
from nomad.config import config
from nomad.parsing.file_parser.text_parser import Quantity as TextQuantity
from nomad.parsing.file_parser.text_parser import TextParser
from nomad.parsing.parser import MatchingParser

from nomad_parser_plugin_boss.schema_packages.schema_package import (
    PotentialEnergySurfaceFit,
    generate_slices,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

configuration = config.get_plugin_entry_point(
    'nomad_parser_plugin_boss.parsers:parser_entry_point'
)


class BossSliceParser(TextParser):
    def init_quantities(self):
        def split_row(full_row: str) -> dict[str, float]:
            split_row = full_row.split()
            key_defs = (
                ('x_1', 'x_2', 'mu', 'nu')
                if len(split_row) == 4
                else ('x_1', 'mu', 'nu')
            )
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


class BossPostProcessingParser(MatchingParser):  # ! TODO: redo
    def parse_datfile(
        self, datfile: str, child_archive: 'EntryArchive', logger: 'BoundLogger'
    ) -> list[float]:
        print(datfile)
        slice_parser = BossSliceParser(mainfile=datfile, logger=logger)
        slice_parser.parse()
        for row in slice_parser.results.get('row', []):
            yield row

    def save_datfile(
        self, row: list[float], logger: 'BoundLogger'
    ) -> PotentialEnergySurfaceFit:
        def get_column_unique(column_name: str) -> list[float]:
            return np.sort(list({x.get(column_name) for x in row}))

        def get_column(column_name: str) -> list[float]:
            return [x.get(column_name) for x in row]

        def reshaping(target: list, dim_1: int, dim_2: int) -> np.ndarray:
            if dim_2:
                return np.reshape(target, (dim_1, dim_2))
            else:
                return np.reshape(target, (dim_1, -1))

        x_1, x_2 = get_column_unique('x_1'), get_column_unique('x_2')

        PotentialEnergySurfaceFit(
            parameter_1_name='parameter_1_name',
            parameter_1_values=x_1,
            parameter_2_name='parameter_2_name',
            parameter_2_values=x_2,
            energy_values=reshaping(get_column('mu'), len(x_1), len(x_2)),
            energy_variance=reshaping(get_column('nu'), len(x_1), len(x_2)),
        )

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        logger.info('BossPostProcessingParser.parse', parameter=configuration.parameter)

        # https://cest-group.gitlab.io/boss/_modules/boss/pp/pp_main.html#PPMain.rstfile
        res = BOResults.from_file(
            mainfile, os.path.join(os.path.dirname(mainfile), 'boss.out')
        )
        iter_no, no_grid_points = (res.settings.get('iterpts', 1), 50)  # ! 250
        pp = PPMain(
            res,
            pp_models=True,
            pp_iters=[iter_no],
            pp_model_slice=[1, 2, no_grid_points],
        )
        bounds = pp.settings.get('bounds', [])

        @staticmethod
        def compute_parameters(rank: int):
            return np.linspace(bounds[rank][0], bounds[rank][1], num=no_grid_points)

        # Set up the archive
        archive.data = PotentialEnergySurfaceFit()

        # Generate slices
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
                X = build_query_points(
                    pp.settings, res.select('x_glmin', iter_no)
                )  # ? change to local minima

                mu, var = res.reconstruct_model(iteration).predict(X)
                mu_all_slices.append(mu.reshape(no_grid_points, no_grid_points))
                var_all_slices.append(var.reshape(no_grid_points, no_grid_points))

            # Save slices
            slice_path = f'parameter_slices/{parameter_counter}'
            section = archive.data.m_setdefault(slice_path)

            section.fit = np.array(mu_all_slices)
            section.uncertainty = np.sqrt(var_all_slices)
            section.iteration = iteration_procedure
            section.parameters_x = np.array(compute_parameters(main_rank))
            section.parameters_y = np.array(compute_parameters(upper_rank))
