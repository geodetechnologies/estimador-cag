import pytest

from evals.stress.scenarios import generate_scenario


PROFILES = ('growing', 'pivot', 'contradiction')
LENGTHS = (1, 3, 6, 10, 20)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('length', LENGTHS)
def test_scenario_has_a_distinct_tracked_fact_for_every_turn(profile, length):
    scenario = generate_scenario(profile, length)

    assert scenario.profile == profile
    assert len(scenario.turns) == length
    assert list(scenario.fact_tracker) == list(range(1, length + 1))
    assert [turn.number for turn in scenario.turns] == list(range(1, length + 1))
    assert len(set(scenario.fact_tracker.values())) == length
    for turn in scenario.turns:
        fact = scenario.fact_tracker[turn.number]
        assert f'Turn {turn.number}:' not in fact
        assert fact in turn.message
        assert 'Synthetic scheduling project' in turn.message
    assert generate_scenario(profile, length) == scenario


@pytest.mark.parametrize('profile', PROFILES)
def test_shorter_conversations_are_prefixes_of_the_same_project(profile):
    full = generate_scenario(profile, 20)
    for length in LENGTHS:
        shorter = generate_scenario(profile, length)
        assert shorter.turns == full.turns[:length]
        assert shorter.fact_tracker == {n: full.fact_tracker[n] for n in range(1, length + 1)}


def test_growing_project_tracks_original_project_name_through_turn_twenty():
    scenario = generate_scenario('growing', 20)
    assert scenario.fact_tracker[1] == 'Harbor'
    assert scenario.fact_tracker[1] in scenario.turns[0].message
    assert all('Harbor' not in turn.message for turn in scenario.turns[1:])
    # A real project_name field contains just the name, not a turn label.
    project_metadata = {'project_name': 'Harbor'}
    assert scenario.fact_tracker[1].casefold() in project_metadata['project_name'].casefold()
    assert 'web portal' in scenario.turns[0].message
    assert 'calendar' in scenario.fact_tracker[3]
    assert 'reminders' in scenario.fact_tracker[5]
    assert 'cloud hosting' in scenario.fact_tracker[6]
    assert 'audit log' in scenario.fact_tracker[8]


def test_pivot_changes_the_technology_stack_at_turn_five():
    scenario = generate_scenario('pivot', 6)
    assert scenario.fact_tracker[1] == 'React'
    assert scenario.fact_tracker[5] == 'Flutter'
    assert 'React' in scenario.turns[0].message
    assert 'React' in scenario.turns[4].message
    assert 'Flutter' in scenario.turns[4].message
    assert 'pivot' in scenario.turns[4].message.lower()
    assert 5 not in generate_scenario('pivot', 3).fact_tracker


def test_contradictions_are_for_the_same_budget_at_turns_three_and_eight():
    scenario = generate_scenario('contradiction', 10)
    assert scenario.fact_tracker[3] == 'presupuesto 30k€'
    assert scenario.fact_tracker[8] == 'presupuesto 80k€'
    assert 'presupuesto 30k€' in scenario.turns[2].message
    assert 'presupuesto 80k€' in scenario.turns[7].message
    assert 8 not in generate_scenario('contradiction', 6).fact_tracker


@pytest.mark.parametrize('profile', ['', 'Growing', 'unknown'])
def test_rejects_unknown_profiles(profile):
    with pytest.raises(ValueError, match='^Unknown scenario profile:'):
        generate_scenario(profile, 1)


@pytest.mark.parametrize('length', [0, 2, 5, 21, True, 1.0])
def test_rejects_unsupported_lengths(length):
    with pytest.raises(ValueError, match='^Unsupported scenario length:'):
        generate_scenario('growing', length)
