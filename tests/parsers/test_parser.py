import json
import shutil

from nomad.client import normalize_all, parse
from nomad.utils import hash as m_hash


def test_dual_entry_creation(upload_dir, synthetic_pes):
    """Parsing the raw .rst creates the referencing raw-file entry and the
    editable measurement .archive.json."""
    archive = parse(str(upload_dir / 'boss.rst'))[0]

    upload_id = None  # client-side parsing has no upload
    assert archive.data.measurement.m_proxy_value == (
        f'../uploads/{upload_id}/archive/'
        f'{m_hash(upload_id, "boss.archive.json")}#data'
    )
    assert (upload_dir / 'boss.archive.json').exists()


def test_dotted_filename_consistency(upload_dir, synthetic_pes):
    """Only the final extension is stripped from the data file name, so the
    child archive and the auxiliary .h5 share the same dotted stem."""
    shutil.copy(upload_dir / 'boss.rst', upload_dir / 'my.run.rst')

    parse(str(upload_dir / 'my.run.rst'))
    assert (upload_dir / 'my.run.archive.json').exists()

    measurement = parse(str(upload_dir / 'my.run.archive.json'))[0]
    normalize_all(measurement)
    assert measurement.data.auxiliary_file == 'my.run.h5'
    assert (upload_dir / 'my.run.h5').exists()


def test_reparse_preserves_edited_child(upload_dir, synthetic_pes):
    """Re-parsing the raw file must not clobber a user-edited measurement entry
    (create_archive is called with overwrite=False)."""
    parse(str(upload_dir / 'boss.rst'))

    child_file = upload_dir / 'boss.archive.json'
    content = json.loads(child_file.read_text())
    content['data']['parameter_names'] = ['phi', 'psi']
    child_file.write_text(json.dumps(content))
    edited_bytes = child_file.read_bytes()

    parse(str(upload_dir / 'boss.rst'))

    assert child_file.read_bytes() == edited_bytes
