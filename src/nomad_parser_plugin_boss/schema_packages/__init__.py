from nomad.config.models.plugins import SchemaPackageEntryPoint


class BOSSSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_parser_plugin_boss.schema_packages.schema_package import m_package

        return m_package


schema_package_entry_point = BOSSSchemaPackageEntryPoint(
    name='BOSS Schema',
    description='Schema package for BOSS Bayesian Optimization analysis FAIR data.',
)
