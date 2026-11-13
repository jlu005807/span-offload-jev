import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preparation = load_script('prepare_appworld_test', ROOT / 'scripts/jev_lora/prepare_appworld_test.py')
_trace = load_script('conversion_fixtures', ROOT / 'tests/test_jev_lora_conversion.py')._trace


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_inputs(tmp_path):
    data = tmp_path / 'data'
    trace = _trace()
    trace_dir = data / 'task-1' / 'original'
    trace_dir.mkdir(parents=True)
    (trace_dir / 'trace.json').write_text(json.dumps(trace), encoding='utf-8')
    manifest = {'tasks': [{'task_id': 'task-1', 'native_split': 'test_normal'}]}
    (data / 'MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
    labels = tmp_path / 'labels.jsonl'
    rows = [{'task_id': 'task-1', 'k': 1, 'label': 2, 'mask': 1,
             'label_version': 'counterfactual-l123-v1', 'reasons': []}]
    labels.write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
    summary = tmp_path / 'summary.json'
    summary.write_text(json.dumps({'n_points': 1, 'n_tasks': 1,
                                  'label_version': 'counterfactual-l123-v1',
                                  'label_distribution': {'2': 1}}), encoding='utf-8')
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({
        'status': 'complete', 'job_id': '667297', 'candidate_points': 1, 'label_rows': 1,
        'tasks_with_points': 1, 'parameters': {'bootstrap': 2000, 'expected_reps': 10,
                                            'seed': 20260920, 'legacy_global_rng': False},
        'data': str(data), 'output': str(tmp_path),
        'input_sha256': {'MANIFEST.json': digest(data / 'MANIFEST.json')},
        'output_sha256': {labels.name: digest(labels), summary.name: digest(summary)},
    }), encoding='utf-8')
    train = tmp_path / 'train.jsonl'
    dev = tmp_path / 'dev.jsonl'
    for path, task in [(train, 'train-task'), (dev, 'dev-task')]:
        path.write_text(json.dumps({'_meta': {'task_id': task}}) + '\n', encoding='utf-8')
    return dict(data=data, labels=labels, label_summary=summary, label_receipt=receipt,
                train_file=train, dev_file=dev, out=tmp_path / 'converted', expected_label_job='667297')


def test_test_conversion_reuses_v1_and_preserves_inputs(tmp_path):
    args = make_inputs(tmp_path)
    before = {name: digest(args[name]) for name in ['labels', 'train_file', 'dev_file']}
    summary = preparation.prepare_test(**args)
    rows = [json.loads(line) for line in (args['out'] / 'test.jsonl').read_text().splitlines()]
    assert rows[0]['labels'] == {'route': 'SLM'}
    assert rows[0]['_meta']['task_id'] == 'task-1'
    assert summary['split'] == 'test'
    assert summary['test']['converted_rows'] == 1
    assert summary['evaluation_only'] is True
    assert before == {name: digest(args[name]) for name in before}
    assert not (args['out'] / 'train.jsonl').exists()
    assert not (args['out'] / 'dev.jsonl').exists()


@pytest.mark.parametrize('failure', ['unfinished', 'tampered_labels', 'duplicate', 'overlap', 'existing_output'])
def test_rejects_bad_upstream_artifacts_without_success_receipt(tmp_path, failure):
    args = make_inputs(tmp_path)
    if failure == 'unfinished':
        receipt = json.loads(args['label_receipt'].read_text())
        receipt['status'] = 'running'
        args['label_receipt'].write_text(json.dumps(receipt))
    elif failure == 'tampered_labels':
        args['labels'].write_text(args['labels'].read_text() + ' ')
    elif failure == 'duplicate':
        args['labels'].write_text(args['labels'].read_text() * 2)
        receipt = json.loads(args['label_receipt'].read_text())
        receipt['output_sha256'][args['labels'].name] = digest(args['labels'])
        args['label_receipt'].write_text(json.dumps(receipt))
    elif failure == 'overlap':
        args['dev_file'].write_text(json.dumps({'_meta': {'task_id': 'task-1'}}) + '\n')
    else:
        args['out'].mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        preparation.prepare_test(**args)
    assert not (args['out'] / 'conversion_summary.json').exists()


def test_loader_acceptance_records_each_rejected_point_and_exact_labels(tmp_path):
    args = make_inputs(tmp_path)
    preparation.prepare_test(**args)

    def loader(tok, path, max_state, max_branch, max_tokens):
        assert (max_state, max_branch, max_tokens) == (4096, 2048, 6144)
        row = json.loads(path.read_text().strip())
        return ([{'enc': {'labels': [1], 'ids': [1, 2, 3]},
                  'meta': [{'type': 'choice', 'keys': ['LLM', 'SLM']}],
                  '_meta': row['_meta']}], 0)

    result = preparation.validate_loader(
        args['out'], tokenizer=object(), load_items=loader,
        model_limits={'max_state': 4096, 'max_branch': 2048},
    )
    assert result['ready_for_evaluation'] is True
    assert result['accepted_rows'] == 1
    assert result['dropped_rows'] == 0
    assert result['label_distribution'] == {'SLM': 1}


def test_oversize_and_masked_rows_keep_exclusion_identity(tmp_path):
    args = make_inputs(tmp_path)
    preparation.prepare_test(**args, max_state_chars=1)
    summary = json.loads((args['out'] / 'conversion_summary.json').read_text())
    assert summary['test']['excluded_online_unavailable'] == 1
    excluded = json.loads((args['out'] / 'conversion_exclusions.jsonl').read_text())
    assert (excluded['task_id'], excluded['k'], excluded['reason']) == ('task-1', 1, 'state_too_large')


def test_loader_retains_identity_when_real_context_limit_drops_a_row(tmp_path):
    args = make_inputs(tmp_path)
    preparation.prepare_test(**args)
    path = args['out'] / 'test.jsonl'
    row = json.loads(path.read_text())
    second = json.loads(json.dumps(row))
    second['_meta']['k'] = 2
    path.write_text(json.dumps(row) + '\n' + json.dumps(second) + '\n')
    summary_path = args['out'] / 'conversion_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['test_sha256'] = digest(path)
    summary_path.write_text(json.dumps(summary))

    def loader(*args, **kwargs):
        return ([{'enc': {'labels': [1], 'ids': [1]},
                  'meta': [{'type': 'choice', 'keys': ['LLM', 'SLM']}],
                  '_meta': row['_meta']}], 1)

    result = preparation.validate_loader(args['out'], tokenizer=None, load_items=loader,
                                         model_limits={'max_state': 4096, 'max_branch': 2048})
    assert result['accepted_rows'] == 1
    assert result['dropped_rows'] == 1
    assert result['exclusions'] == [{'task_id': 'task-1', 'k': 2, 'reason': 'openjev_loader_rejected'}]
