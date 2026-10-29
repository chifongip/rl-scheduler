"""Exercise the dashboard's real log readers with controlled concurrent responses."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js is required for dashboard checks')
def test_log_reads_are_serialized_and_preserve_view_state():
    html = (Path(__file__).resolve().parents[1] / 'static' / 'index.html').read_text()
    readers = html[html.index('async function loadLogTail()'):html.index('function closeLogModal()')]
    harness = r'''
const assert = require('node:assert/strict');
const API = '';
let logState;
const elements = {
    'log-content': {textContent: 'existing', scrollTop: 120, scrollHeight: 1000, clientHeight: 100},
    'log-autoscroll': {checked: false},
    'log-notice': {textContent: '', classList: {toggle(){}, remove(){}}},
};
const document = {getElementById: id => elements[id]};
function markAdminActivity() {}
const requests = [];
function fetch(path) {
    return new Promise(resolve => requests.push({path, resolve}));
}
function respond(index, text, offset = '10') {
    requests[index].resolve({ok: true, text: async () => text,
        headers: {get: name => name === 'X-Log-Next-Offset' ? offset :
            name === 'X-Log-Size' ? offset : null}});
}
'''
    scenarios = r'''
(async () => {
    logState = {kind: 'task', taskId: 'task', offset: 0};
    const incremental = refreshOpenLog();
    await Promise.all([refreshOpenLog(), loadLogTail()]);
    assert.equal(requests.length, 1, 'polling and manual refresh must share one pending read');
    respond(0, ' chunk');
    await incremental;
    assert.equal(elements['log-content'].textContent, 'existing chunk');
    assert.equal(logState.offset, 10);
    assert.equal(logState.loading, false);

    logState = {kind: 'tensorboard', path: '/tensorboards/test/logs', user: 'alice', loaded: true};
    const snapshot = loadLogTail();
    await Promise.all([refreshOpenLog(), loadLogTail()]);
    assert.equal(requests.length, 2, 'snapshot reads must also be serialized');
    respond(1, 'snapshot');
    await snapshot;
    assert.equal(elements['log-content'].textContent, 'snapshot');
    assert.equal(elements['log-content'].scrollTop, 120, 'manual scroll position must survive snapshots');

    const late = loadLogTail();
    logState = {kind: 'task', taskId: 'other', offset: 0};
    respond(2, 'obsolete');
    await late;
    assert.equal(elements['log-content'].textContent, 'snapshot', 'late responses must not replace a different log');

    // Rotation must release the reader before reloading the tail.
    logState.offset = 100;
    const rotated = refreshOpenLog();
    respond(3, '', '0');
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(requests.length, 5);
    assert.match(requests[4].path, /tail_bytes=/);
    respond(4, 'rotated log');
    await rotated;
    assert.equal(elements['log-content'].textContent, 'rotated log');
    assert.equal(logState.loading, false);
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run(
        [shutil.which('node')], input=harness + readers + scenarios,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
