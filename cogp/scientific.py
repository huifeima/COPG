"""Fail-closed provenance requirements for fixed-protocol scientific runs.

These checks bind declared identity and completion metadata to registered
checkpoint bytes. They cannot prove that training occurred or make rebuilt
models numerically equivalent to the historical experiment implementation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from .io import digest


def _mapping(value, label):
    if not isinstance(value, Mapping) or not value:
        raise ValueError('Scientific checkpoint requires explicit ' + label)
    return value


def _integer(value, expected, label):
    if type(value) is not int or value != expected:
        raise ValueError(f'Scientific checkpoint requires {label}={expected}')


def _checksum(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Scientific checkpoint requires SHA-256 for ' + label)


def _config(value, config, label):
    value = _mapping(value, label)
    expected = {field.name for field in fields(config)}
    if set(value) != expected:
        raise ValueError('Scientific checkpoint requires complete ' + label +
                         f'; missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}')
    if dict(value) != config.to_dict():
        raise ValueError('Scientific checkpoint configuration differs from loaded ' + label)


def validate_parent(loaded, *, seed, receipts):
    """Require explicit identity/configuration and a byte-bound final receipt."""
    from .learning.parent.checkpoint import ARCHIVED_METHODS
    payload = loaded.payload
    if payload.get('test_only') or payload.get('job', {}).get('test_only'):
        raise ValueError('Engineering-test checkpoint cannot be used in a scientific panel')
    job = _mapping(payload.get('job'), 'parent job')
    learner = _mapping(payload.get('learner'), 'parent learner')
    _integer(job.get('seed'), seed, 'parent training seed')
    _integer(learner.get('decisions'), 32768, 'parent joint decisions')
    if ARCHIVED_METHODS.get(job.get('method'), job.get('method')) != loaded.method:
        raise ValueError('Scientific parent job must declare its allocation method')
    for label, config in (('env_cfg', loaded.env_cfg), ('planner_cfg', loaded.planner_cfg)):
        _config(learner.get(label), config, label)
        if label in job and dict(_mapping(job[label], 'job ' + label)) != dict(learner[label]):
            raise ValueError('Parent job and learner configurations disagree: ' + label)
    _mapping(job.get('protocol'), 'parent training protocol')
    _checksum(job.get('study_hash'), 'parent study identity')
    basc = _mapping(job.get('basc_sha256'), 'frozen BASC training asset identity')
    for name in ('collect', 'drive'):
        _checksum(basc.get(name), 'training BASC ' + name)
    matches = [r for r in receipts if str(r.get('status', '')).upper() == 'COMPLETE'
               and r.get('checkpoint_sha256') == loaded.checkpoint_sha256]
    if not matches:
        raise ValueError('Scientific parent requires a registered checkpoint-bound completion receipt')
    for receipt in matches:
        if receipt.get('test_only'):
            raise ValueError('Engineering-test checkpoint receipt cannot be used in a scientific panel')
        _integer(receipt.get('seed'), seed, 'parent receipt seed')
        _integer(receipt.get('decisions'), 32768, 'parent receipt joint decisions')
        if (ARCHIVED_METHODS.get(receipt.get('method'), receipt.get('method')) != loaded.method
                or receipt.get('study_hash') != job['study_hash']):
            raise ValueError('Parent completion receipt conflicts with checkpoint identity')


def validate_dynamics(model, *, seed, kind, receipts):
    """A conversion proof alone is not evidence of completed model fitting."""
    identity = _mapping(model.checkpoint_identity, 'dynamics training identity')
    if identity.get('test_only'):
        raise ValueError('Engineering-test dynamics identity cannot be used in a scientific panel')
    _integer(identity.get('seed'), seed, 'dynamics training seed')
    _integer(identity.get('updates'), 5000, 'dynamics optimizer updates')
    if identity.get('kind') != kind or dict(_mapping(identity.get('geometry'), 'dynamics training geometry')) != model.config:
        raise ValueError('Dynamics training identity conflicts with model kind or physical scales')
    _checksum(identity.get('dataset_sha256'), 'dynamics training dataset')
    for key in ('batch_size', 'learning_rate', 'weight_decay', 'loss', 'split'):
        if key not in identity:
            raise ValueError('Scientific dynamics checkpoint requires training configuration: ' + key)
    identity_hash = digest(identity)
    if getattr(model, 'checkpoint_identity_sha256', None) != identity_hash:
        raise ValueError('Dynamics checkpoint identity digest is missing or inconsistent')
    matches = [r for r in receipts if str(r.get('status', '')).upper() == 'COMPLETE'
               and r.get('checkpoint_sha256') == model.checkpoint_sha256]
    if not matches:
        raise ValueError('Scientific dynamics requires a registered checkpoint-bound fit completion report')
    for receipt in matches:
        if receipt.get('test_only'):
            raise ValueError('Engineering-test dynamics report cannot be used in a scientific panel')
        _integer(receipt.get('optimizer_updates'), 5000, 'dynamics receipt optimizer updates')
        if receipt.get('identity') != dict(identity) or receipt.get('identity_sha256') != identity_hash:
            raise ValueError('Dynamics completion report conflicts with checkpoint training identity')
