#!/usr/bin/env python3
"""校验已完成的 test 标注，复用冻结 V1 转换器；单独验收 OpenJev loader。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    'appworld_v1_converter', Path(__file__).with_name('convert_appworld_labels.py'))
converter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(converter)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path: Path, data: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def prepare_test(*, data: Path, labels: Path, label_summary: Path, label_receipt: Path,
                 train_file: Path, dev_file: Path, out: Path, expected_label_job: str,
                 max_state_chars: int = 12000) -> dict:
    """仅生成独立 test.jsonl，不重写 train/dev，也不执行训练或模型推理。"""
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite conversion output: {out}')
    receipt = json.loads(label_receipt.read_text(encoding='utf-8'))
    if receipt.get('status') != 'complete' or receipt.get('job_id') != expected_label_job:
        raise ValueError('Upstream label job has no matching complete receipt')
    if receipt.get('parameters') != dict(bootstrap=2000, expected_reps=10,
                                        seed=20260920, legacy_global_rng=False):
        raise ValueError('Label parameters differ from the frozen train/dev protocol')
    if Path(receipt['data']).resolve() != data.resolve() or Path(receipt['output']).resolve() != labels.parent.resolve():
        raise ValueError('Label receipt paths do not match conversion inputs')
    for name, expected in receipt['input_sha256'].items():
        if digest(data / name) != expected:
            raise ValueError(f'Input manifest changed: {name}')
    for path in [labels, label_summary]:
        if digest(path) != receipt['output_sha256'].get(path.name):
            raise ValueError(f'Upstream label artifact hash mismatch: {path}')
    manifest = json.loads((data / 'MANIFEST.json').read_text(encoding='utf-8'))
    tasks = {row['task_id']: row['native_split'] for row in manifest['tasks']}
    if len(tasks) != len(manifest['tasks']) or not set(tasks.values()) <= {'test_normal', 'test_challenge'}:
        raise ValueError('Expected unique official test task IDs in manifest')
    rows = read_rows(labels)
    pairs = [(row['task_id'], row['k']) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError('Duplicate task/k in labels')
    if len(rows) != receipt['candidate_points'] or len(rows) != receipt['label_rows']:
        raise ValueError('Label coverage does not match complete receipt')
    summary = json.loads(label_summary.read_text(encoding='utf-8'))
    distribution = dict(Counter(str(row['label']) for row in rows))
    if (summary['n_points'] != len(rows) or summary['n_tasks'] != len({t for t, k in pairs})
            or summary['label_distribution'] != distribution):
        raise ValueError('Label summary does not match raw rows')
    for row in rows:
        if (row['task_id'] not in tasks or type(row['k']) is not int or row['k'] < 1
                or row.get('label_version') != 'counterfactual-l123-v1'):
            raise ValueError('Invalid test label identity/version')
        if not ((type(row.get('mask')) is int and row['mask'] == 0 and row['label'] == 'unknown')
                or (type(row.get('mask')) is int and row['mask'] == 1
                    and type(row['label']) is int and row['label'] in (0, 1, 2))):
            raise ValueError('Invalid label/mask pair')
    reference_hashes = {}
    for split, path in [('train', train_file), ('dev', dev_file)]:
        reference_tasks = {r['_meta']['task_id'] for r in read_rows(path)}
        overlap = reference_tasks & set(tasks)
        if overlap:
            raise ValueError(f'Test overlaps {split}: {sorted(overlap)}')
        reference_hashes[split] = digest(path)
    if converter.v1_prompt_sha256() != converter.EXPECTED_V1_PROMPT_SHA256:
        raise ValueError('V1 prompt changed')

    converted, excluded = [], []
    cached_task, trace = None, None
    for row in sorted(rows, key=lambda r: (r['task_id'], r['k'])):
        identity = {'task_id': row['task_id'], 'k': row['k'], 'label': row['label']}
        if row['mask'] == 0:
            excluded.append({**identity, 'reason': 'unknown_or_masked', 'reasons': row.get('reasons', [])})
            continue
        if row['task_id'] != cached_task:
            cached_task = row['task_id']
            trace = json.loads((data / cached_task / 'original/trace.json').read_text(encoding='utf-8'))
        try:
            item = converter.convert_label_row(row, trace, max_history_spans=10,
                                               max_state_chars=max_state_chars, request_model_id='wev-1.7b')
        except converter.ConversionExcluded as error:
            excluded.append({**identity, 'reason': error.reason})
            continue
        if (item['request']['state']['model_info']['large_model'],
                item['request']['state']['model_info']['small_model']) != ('Qwen3.8-27B', 'Qwen3.5-9B'):
            raise ValueError('Test trace model pair differs from train/dev')
        converted.append(item)

    exclusions = Counter(row['reason'] for row in excluded)
    result = {
        'split': 'test', 'evaluation_only': True, 'format': 'OpenJev System One JSONL',
        'prompt_version': 'v1', 'prompt_sha256': converter.v1_prompt_sha256(),
        'input_mode': 'context', 'request_model_id': 'wev-1.7b',
        'max_history_spans': 10, 'max_state_chars': max_state_chars,
        'label_mapping': {'0': 'LLM', '1': 'SLM', '2': 'SLM', 'unknown': 'excluded'},
        'label_job': expected_label_job, 'manifest_task_count': len(tasks),
        'test': {'label_rows': len(rows), 'converted_rows': len(converted),
                 'task_count': len({r['_meta']['task_id'] for r in converted}),
                 'excluded_unknown_or_masked': exclusions.get('unknown_or_masked', 0),
                 'excluded_online_unavailable': len(excluded) - exclusions.get('unknown_or_masked', 0),
                 'excluded_by_reason': dict(exclusions),
                 'label_distribution': dict(Counter(r['labels']['route'] for r in converted))},
        'train_dev_task_overlap': 0, 'train_dev_sha256': reference_hashes,
        'input_sha256': {str(p): digest(p) for p in [labels, label_summary, label_receipt, data / 'MANIFEST.json']},
        'source_sha256': {p.name: digest(p) for p in [Path(__file__), Path(converter.__file__)]},
    }
    out.mkdir(parents=True, exist_ok=False)
    for filename, items in [('test.jsonl', converted), ('conversion_exclusions.jsonl', excluded)]:
        with (out / filename).open('x', encoding='utf-8') as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')
    result['test_sha256'] = digest(out / 'test.jsonl')
    write_json(out / 'conversion_summary.json', result)
    return result


def validate_loader(directory: Path, *, tokenizer, load_items, model_limits: dict,
                    max_tokens: int = 6144) -> dict:
    """逐条运行真实 loader，仅编码不加载权重；保留被拒绝点身份并核对 label。"""
    summary = json.loads((directory / 'conversion_summary.json').read_text(encoding='utf-8'))
    path = directory / 'test.jsonl'
    if digest(path) != summary['test_sha256']:
        raise ValueError('Converted test data changed before loader acceptance')
    rows = read_rows(path)
    items, dropped = load_items(tokenizer, path, model_limits['max_state'],
                                model_limits['max_branch'], max_tokens=max_tokens)
    by_key = {(r['_meta']['task_id'], r['_meta']['k']): r for r in rows}
    accepted = set()
    distribution = Counter()
    token_lengths = []
    for item in items:
        key = (item['_meta']['task_id'], item['_meta']['k'])
        if key not in by_key or key in accepted:
            raise ValueError('Unexpected or duplicate loader item identity')
        meta, gold = item['meta'], item['enc']['labels']
        if (len(meta) != 1 or len(gold) != 1 or meta[0]['type'] != 'choice'
                or meta[0]['keys'] != ['LLM', 'SLM']
                or type(gold[0]) is not int or gold[0] not in (0, 1)):
            raise ValueError('Loader did not produce one binary route choice')
        route = meta[0]['keys'][gold[0]]
        if route != by_key[key]['labels']['route']:
            raise ValueError('Loader label differs from converted gold route')
        distribution[route] += 1
        token_lengths.append(len(item['enc']['ids']))
        accepted.add(key)
    if len(items) + dropped != len(rows) or not items:
        raise ValueError('Loader row accounting failed or no test samples accepted')
    excluded = [{'task_id': t, 'k': k, 'reason': 'openjev_loader_rejected'}
                for t, k in by_key if (t, k) not in accepted]
    if len(excluded) != dropped:
        raise ValueError('Dropped count differs from excluded point identities')
    result = {'ready_for_evaluation': True, 'split': 'test', 'converted_rows': len(rows),
              'accepted_rows': len(items), 'dropped_rows': dropped,
              'label_distribution': dict(distribution), 'max_tokens': max_tokens,
              'model_limits': model_limits, 'max_observed_input_tokens': max(token_lengths),
              'tokenizer_sha256': tokenizer_sha256(tokenizer),
              'test_sha256': digest(path), 'exclusions': excluded}
    write_json(directory / 'loader_acceptance.json', result)
    return result


def tokenizer_sha256(tokenizer) -> str | None:
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    serialize = getattr(backend, 'to_str', None)
    if not callable(serialize):
        return None
    return hashlib.sha256(serialize().encode('utf-8')).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--label-summary', type=Path)
    parser.add_argument('--label-receipt', type=Path)
    parser.add_argument('--train-file', type=Path)
    parser.add_argument('--dev-file', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--expected-label-job')
    parser.add_argument('--validate-loader', action='store_true')
    parser.add_argument('--model-run', type=Path)
    args = parser.parse_args(argv)
    if args.validate_loader:
        if not args.model_run:
            parser.error('--model-run is required for loader acceptance')
        import torch
        from transformers import AutoTokenizer
        from wev.data import load_items
        meta = torch.load(args.model_run / 'head.pt', map_location='cpu', weights_only=False)
        tok = AutoTokenizer.from_pretrained(meta['base'], revision=meta.get('base_revision'), local_files_only=True)
        limits = {'max_state': meta['max_state'], 'max_branch': meta['max_branch']}
        result = validate_loader(args.out, tokenizer=tok, load_items=load_items, model_limits=limits)
        result['model_run'] = str(args.model_run)
        result['head_sha256'] = digest(args.model_run / 'head.pt')
        result['adapter_sha256'] = digest(args.model_run / 'adapter_model.safetensors')
        result['base_model'] = meta['base']
        if result['tokenizer_sha256'] is None:
            raise ValueError('Could not fingerprint the tokenizer loaded by OpenJev')
        # Keep model identity in its own receipt; loader_acceptance is immutable.
        write_json(args.out / 'loader_model_identity.json', {
            'model_run': str(args.model_run), 'head_sha256': result['head_sha256'],
            'adapter_sha256': result['adapter_sha256'], 'base': meta['base'],
            'tokenizer_sha256': result['tokenizer_sha256'],
        })
        print(json.dumps({k: v for k, v in result.items() if k != 'exclusions'}, ensure_ascii=False, indent=2))
        return 0
    else:
        names = ['data', 'labels', 'label_summary', 'label_receipt', 'train_file', 'dev_file', 'expected_label_job']
        if any(getattr(args, name) is None for name in names):
            parser.error('All label, trace and train/dev reference arguments are required')
        result = prepare_test(**{name: getattr(args, name) for name in names}, out=args.out)
    print(json.dumps({k: v for k, v in result.items() if k != 'exclusions'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
