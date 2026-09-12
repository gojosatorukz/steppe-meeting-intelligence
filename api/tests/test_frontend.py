"""Integration checks with the model boundary stubbed only inside tests."""
import copy
from pathlib import Path
from fastapi.testclient import TestClient
import main
import exports

SAMPLE = {'transcript': 'Текст < 10 & обсуждение.', 'language': 'ru', 'audio_seconds': 2.0,
          'segments': [], 'protocol': {'summary': ['Первое.', 'Второе.', 'Третье.'],
          'decisions': ['Бюджет < 100 & резерв'], 'topics': [], 'open_questions': [],
          'action_items': [{'owner': 'не назначен', 'task': 'Проверить <A>', 'due': 'к пятнице', 'priority': 'высокий'}], 'risks': []},
          'timing': {'total_sec': 1.0, 'transcribe_sec': 0.5, 'llm_sec': 0.5}}

def test_frontend_routes_and_local_assets():
    client = TestClient(main.app)
    page = client.get('/')
    assert page.status_code == 200
    assert 'src="/assets/' in page.text
    assert 'steppe' in page.text.lower()
    assert client.get('/live').status_code == 200
    assert client.get('/legacy').status_code == 200
    assert client.get('/legacy-live').status_code == 200
    assert client.get('/favicon.svg').status_code == 200
    import re
    for url in re.findall(r'(?:src|href)="(/assets/[^"]+)"', page.text):
        assert client.get(url).status_code == 200

def test_existing_process_contract_and_exports(monkeypatch):
    monkeypatch.setattr(main, 'RESULTS', {})
    monkeypatch.setattr(main.pipeline, 'process_audio', lambda path, **kw: copy.deepcopy(SAMPLE))
    monkeypatch.setattr(main.pipeline, 'process_text', lambda transcript, **kw: copy.deepcopy(SAMPLE))
    client = TestClient(main.app)
    a = client.post('/process', files={'file': ('sample.wav', b'unit-test-only', 'audio/wav')})
    assert a.status_code == 200
    rid = a.json()['id']
    assert client.get('/results').json()[0]['id'] == rid
    assert client.get(f'/result/{rid}').json()['protocol']['action_items'][0]['due'] == 'к пятнице'
    assert client.post('/process-text', json={'transcript': 'test-only'}).status_code == 200
    assert client.get(f'/result/{rid}/export.json').status_code == 200
    assert client.get(f'/result/{rid}/export.csv').content.startswith(b'\xef\xbb\xbf')
    pdf = client.get(f'/result/{rid}/export.pdf')
    assert pdf.status_code == 200 and pdf.content.startswith(b'%PDF')
    assert client.get(f'/result/{rid}/audio').status_code == 404
    assert client.post('/process', files={'file': ('bad.exe', b'x')}).status_code == 400

def test_pipeline_failure_is_not_a_success(monkeypatch):
    monkeypatch.setattr(main, 'RESULTS', {})
    def fail(path, **kw):
        raise RuntimeError('Model missing')
    monkeypatch.setattr(main.pipeline, 'process_audio', fail)
    client = TestClient(main.app)
    res = client.post('/process', files={'file': ('sample.wav', b'x')})
    assert res.status_code == 500
    assert 'Model missing' in res.json()['detail']
    assert client.get('/results').json() == []
