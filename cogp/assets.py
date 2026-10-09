"""Explicit checkpoint assets; paths never activate an external source tree."""
from __future__ import annotations
from pathlib import Path
import json
import shutil
from .io import sha, atomic_json


class AssetRegistry:
    def __init__(self, manifest):
        self.path = Path(manifest).expanduser().resolve()
        self.root = self.path.parent
        self.data = json.loads(self.path.read_text())
        self.entries = self.data.get('assets', {})
        if not isinstance(self.entries, dict):
            raise ValueError('Asset manifest must contain an assets mapping')

    def resolve(self, key, verify=True, *, require_sha=False):
        if key not in self.entries:
            raise FileNotFoundError('Missing asset entry: ' + key)
        item = self.entries[key]
        path = Path(item['path']).expanduser()
        if not path.is_absolute():
            path = self.root / path
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint {key}: {path}. Add the real asset; no random-policy fallback.')
        expected = item.get('sha256')
        if require_sha and (not isinstance(expected, str) or len(expected) != 64
                            or any(c not in '0123456789abcdef' for c in expected)):
            raise ValueError('Scientific asset requires an explicit SHA-256: ' + key)
        if verify and expected and sha(path) != expected:
            raise RuntimeError('Checkpoint SHA-256 mismatch: ' + key)
        if verify:
            for companion in item.get('companions', []):
                companion_path=self.root/companion['path']
                if not companion_path.is_file():raise FileNotFoundError(f'Missing companion for {key}: {companion_path}')
                if sha(companion_path)!=companion['sha256']:raise RuntimeError('Asset companion SHA-256 mismatch: '+key)
        return path

    def verified_receipts(self, key):
        """Read registered JSON companions after verifying all asset hashes.

        Unregistered neighbouring files are not scientific training receipts.
        Register the actual producer receipt; this does not authenticate its
        contents or establish historical implementation equivalence.
        """
        self.resolve(key, require_sha=True)
        receipts = []
        for companion in self.entries[key].get('companions', []):
            path = self.root / companion['path']
            if path.suffix == '.json':
                value = json.loads(path.read_text())
                if not isinstance(value, dict):
                    raise ValueError('Asset JSON companion must be a mapping: ' + key)
                receipts.append(value)
        return receipts

    def check(self, required=()):
        report = {'available': [], 'missing': [], 'mismatch': []}
        for key in required or sorted(self.entries):
            try:
                self.resolve(key)
                report['available'].append(key)
            except FileNotFoundError as exc:
                report['missing'].append({'key': key, 'reason': str(exc)})
            except RuntimeError as exc:
                report['mismatch'].append({'key': key, 'reason': str(exc)})
        report['ready'] = not report['missing'] and not report['mismatch']
        return report

    def add(self, key, source, destination=None):
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = Path(destination or self.entries.get(key, {}).get('path', source.name))
        if destination.is_absolute() or '..' in destination.parts:
            raise ValueError('Imported assets must remain inside the asset directory')
        target = self.root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != source:
            if target.exists() and sha(target) != sha(source):
                raise RuntimeError('Different checkpoint already occupies destination: ' + str(target))
            shutil.copy2(source, target)
        companions=[]
        for suffix in ('.json','.report.json','.conversion.json'):
            companion=source.with_suffix(suffix)
            if companion.is_file() and companion!=source:
                companion_target=target.with_suffix(suffix)
                if companion_target.resolve()!=companion:
                    if companion_target.exists() and sha(companion_target)!=sha(companion):raise RuntimeError('Different checkpoint companion already exists: '+str(companion_target))
                    shutil.copy2(companion,companion_target)
                companions.append({'path':str(companion_target.relative_to(self.root)),'sha256':sha(companion_target)})
        self.entries[key] = {'path': str(destination), 'sha256': sha(target), 'companions':companions}
        self.data['assets'] = self.entries
        atomic_json(self.path, self.data)
        return self.entries[key]

