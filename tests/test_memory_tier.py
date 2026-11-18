from copy import deepcopy

import pytest

from app.memory.tier import Tier, resolve_tier
from app.sessions import ProjectMetadata


@pytest.fixture
def resolve(capsys):
    def call(transcript='', *, metadata=None, override=None):
        if metadata is None:
            metadata = ProjectMetadata()
        before = deepcopy(metadata.model_dump())
        result = resolve_tier(transcript=transcript, metadata=metadata, override=override)
        assert metadata.model_dump() == before
        output = capsys.readouterr()
        assert output.out == ''
        assert output.err == ''
        return result

    return call


def test_tier_values():
    assert {tier.name: tier.value for tier in Tier} == {
        'EXECUTIVE': 'executive',
        'PM': 'pm',
        'DEVELOPER': 'developer',
        'DEFAULT': 'default',
    }
    assert isinstance(Tier.EXECUTIVE, str)


@pytest.mark.parametrize('override', list(Tier))
def test_override_wins_every_rule(resolve, override):
    metadata = ProjectMetadata(assumed_team_size=1, agreed_scope=['NDA'])
    assert resolve('GDPR docker kubernetes', metadata=metadata, override=override) == (
        override, 'explicit_override'
    )


@pytest.mark.parametrize('keyword', [
    'nda', 'NDA', 'non-disclosure', 'non disclosure', 'nondisclosure',
    'confidential', 'under embargo', 'legal hold',
])
@pytest.mark.parametrize('location', ['transcript', 'scope'])
def test_nda_keywords_and_precedence(resolve, keyword, location):
    metadata = ProjectMetadata(
        assumed_team_size=2,
        agreed_scope=[keyword if location == 'scope' else 'ordinary scope'],
        mentioned_technologies=['GDPR'],
    )
    transcript = f'{keyword if location == "transcript" else "ordinary"} docker kafka'
    assert resolve(transcript, metadata=metadata) == (Tier.EXECUTIVE, 'nda_detected')


@pytest.mark.parametrize('keyword', [
    'hipaa', 'GDPR', 'sox', 'pci-dss', 'pci dss', 'pcidss', 'fda',
    'iso-27001', 'iso 27001', 'iso27001', 'ccpa',
])
@pytest.mark.parametrize('location', ['transcript', 'scope', 'technology'])
def test_regulatory_keywords_and_precedence(resolve, keyword, location):
    metadata = ProjectMetadata(
        assumed_team_size=2,
        agreed_scope=[keyword] if location == 'scope' else [],
        mentioned_technologies=[keyword] if location == 'technology' else [],
    )
    transcript = f'{keyword if location == "transcript" else "ordinary"} docker kafka'
    assert resolve(transcript, metadata=metadata) == (Tier.EXECUTIVE, 'regulatory_context')


@pytest.mark.parametrize('keyword', [
    'docker', 'kubernetes', 'k8s', 'microservice', 'microservices', 'terraform',
    'iac', 'helm', 'grpc', 'graphql', 'kafka', 'airflow', 'spark', 'rabbitmq',
])
def test_each_technical_keyword_counts_with_another(resolve, keyword):
    other = 'kafka' if keyword != 'kafka' else 'docker'
    metadata = ProjectMetadata(assumed_team_size=2)
    assert resolve(f'{keyword.upper()} {other}', metadata=metadata) == (
        Tier.DEVELOPER, 'technical_audience'
    )


def test_repeated_keyword_does_not_count_twice(resolve):
    assert resolve('Docker DOCKER docker') == (Tier.DEFAULT, 'default')
    assert resolve('Docker DOCKER docker', metadata=ProjectMetadata(assumed_team_size=2)) == (
        Tier.PM, 'low_budget_pm'
    )


def test_technical_keywords_only_come_from_transcript(resolve):
    metadata = ProjectMetadata(agreed_scope=['docker'], mentioned_technologies=['kafka'])
    assert resolve('docker', metadata=metadata) == (Tier.DEFAULT, 'default')
    assert resolve('', metadata=metadata) == (Tier.DEFAULT, 'default')


def test_nda_does_not_use_technologies(resolve):
    metadata = ProjectMetadata(mentioned_technologies=['nda'])
    assert resolve('', metadata=metadata) == (Tier.DEFAULT, 'default')


@pytest.mark.parametrize('size, expected', [
    (1, (Tier.PM, 'low_budget_pm')),
    (2, (Tier.PM, 'low_budget_pm')),
    (3, (Tier.DEFAULT, 'default')),
    (None, (Tier.DEFAULT, 'default')),
])
def test_previous_metadata_team_size_boundary(resolve, size, expected):
    assert resolve('Our team has 100 members', metadata=ProjectMetadata(assumed_team_size=size)) == expected


def test_transcript_team_size_does_not_replace_previous_metadata(resolve):
    assert resolve('Our team has one person') == (Tier.DEFAULT, 'default')
    assert resolve('Our team has one person', metadata=ProjectMetadata(assumed_team_size=3)) == (
        Tier.DEFAULT, 'default'
    )


@pytest.mark.parametrize('transcript', [
    'panda', 'confidentiality', 'under embargoed', 'legal holder',
    'hipaax', 'xgdpr', 'pci-dss2', 'iso-270010',
    'dockers kafka', 'docker kafkacluster',
])
def test_keyword_word_boundaries(resolve, transcript):
    assert resolve(transcript) == (Tier.DEFAULT, 'default')


def test_list_entries_are_independent_not_joined_into_keywords(resolve):
    metadata = ProjectMetadata(
        agreed_scope=['non', 'disclosure', 'legal', 'hold', 'pci', 'dss'],
        mentioned_technologies=['iso', '27001'],
    )
    assert resolve('', metadata=metadata) == (Tier.DEFAULT, 'default')


def test_unrelated_metadata_fields_do_not_affect_resolution(resolve):
    metadata = ProjectMetadata(
        project_name='NDA', explicit_constraints=['GDPR'], rejected_options=['confidential']
    )
    assert resolve('', metadata=metadata) == (Tier.DEFAULT, 'default')
