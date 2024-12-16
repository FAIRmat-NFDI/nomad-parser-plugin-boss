from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from structlog.stdlib import (
        BoundLogger,
    )

import os
import numpy as np

from boss.bo.results import BOResults
from boss.io.dump import build_query_points
from boss.pp.pp_main import PPMain
from nomad.config import config
from nomad.datamodel.datamodel import EntryArchive
from nomad.datamodel.metainfo.annotations import H5WebAnnotation
from nomad.parsing.file_parser.text_parser import Quantity as TextQuantity
from nomad.parsing.file_parser.text_parser import TextParser
from nomad.parsing.parser import MatchingParser

from nomad_parser_plugin_boss.schema_packages.schema_package import (
    PotentialEnergySurfaceFit,
)

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
        archive: EntryArchive,
        logger: 'BoundLogger',
        child_archives: dict[str, EntryArchive] = None,
    ) -> None:
        logger.info('BossPostProcessingParser.parse', parameter=configuration.parameter)

        # https://cest-group.gitlab.io/boss/_modules/boss/pp/pp_main.html#PPMain.rstfile
        res = BOResults.from_file(
            mainfile, os.path.join(os.path.dirname(mainfile), 'boss.out')
        )
        iter_no, no_grid_points = (
            res.settings.get('iterpts', 1),
            50,
        )  # ! make more robust, 250
        pp = PPMain(
            res,
            pp_models=True,
            pp_iters=[iter_no],
            pp_model_slice=[1, 2, no_grid_points],
        )
        bounds = pp.settings.get('bounds', [])
        ranks = len(bounds)

        @staticmethod
        def generate_slices():
            for main_rank in range(ranks):
                for upper_rank in range(main_rank + 1, ranks):
                    yield main_rank, upper_rank

        @staticmethod
        def compute_parameters(rank: int):
            return np.linspace(bounds[rank][0], bounds[rank][1], num=no_grid_points)

        # systematically produce slices over the whole dimension space
        all_parameter_1, all_parameter_2 = map(
            list,
            map(
                lambda f, x: map(f, x),
                (compute_parameters, compute_parameters),
                zip(*generate_slices()),
            ),
        )

        mu_all_slices, var_all_slices = [], []
        for iteration in range(iter_no):
            mu_all_slices.append([])
            var_all_slices.append([])

            for main_rank, upper_rank in generate_slices():
                pp = PPMain(
                    res,
                    pp_models=True,
                    pp_iters=[iteration + 1],
                    pp_model_slice=[main_rank + 1, upper_rank + 1, no_grid_points],
                )
                X = build_query_points(
                    pp.settings, res.select('x_glmin', iter_no)
                )  # ? change to local minima
                mu, var = res.reconstruct_model(iteration + 1).predict(X)
                mu_all_slices[iteration].append(
                    mu.reshape(no_grid_points, no_grid_points)
                )
                var_all_slices[iteration].append(
                    var.reshape(no_grid_points, no_grid_points)
                )

        # Write to archive
        archive.data = PotentialEnergySurfaceFit()

        section = archive.data.m_setdefault('energy_values')
        section.signal = np.array(mu_all_slices)
        section.parameter_1 = np.array(all_parameter_1)
        section.parameter_2 = np.array(all_parameter_2)

        section = archive.data.m_setdefault('energy_std')
        section.signal = np.sqrt(var_all_slices)
        section.parameter_1 = np.array(all_parameter_1)
        section.parameter_2 = np.array(all_parameter_2)
