const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const {
    buildCodexAppServerArgs,
    buildCodexArgs,
    createWavBuffer,
    loadReferenceContext,
    normalizeReferenceMarkdown,
    parseCodexEvent,
    selectReferenceContext,
} = require('../src/utils/gigaam-codex');

test('createWavBuffer produces a 16 kHz mono PCM WAV', () => {
    const pcm = Buffer.alloc(32000);
    const wav = createWavBuffer(pcm);

    assert.equal(wav.toString('ascii', 0, 4), 'RIFF');
    assert.equal(wav.toString('ascii', 8, 12), 'WAVE');
    assert.equal(wav.readUInt16LE(22), 1);
    assert.equal(wav.readUInt32LE(24), 16000);
    assert.equal(wav.readUInt16LE(34), 16);
    assert.equal(wav.readUInt32LE(40), pcm.length);
    assert.equal(wav.length, pcm.length + 44);
});

test('parseCodexEvent returns only completed assistant messages', () => {
    const completed = JSON.stringify({
        type: 'item.completed',
        item: { type: 'agent_message', text: 'answer' },
    });

    assert.equal(parseCodexEvent(completed), 'answer');
    assert.equal(parseCodexEvent('{"type":"turn.started"}'), '');
    assert.equal(parseCodexEvent('not-json'), '');
});

test('normalizeReferenceMarkdown restores URL-encoded formulas', () => {
    const markdown =
        'Ошибка [URL 🔗](https://saxarona.github.io/mathjax-viewer/?input=k_p%5E2%3D4%7B%2C%7D9%5Ccdot10%5E%7B-5%7D#0), k\uE000 и i\uE000.';
    const normalized = normalizeReferenceMarkdown(markdown);

    assert.match(normalized, /\\\[\s*k_p\^2=4\{,\}9\\cdot10\^\{-5\}\s*\\\]/);
    assert.match(normalized, /k_p и i_p\./);
    assert.doesNotMatch(normalized, /mathjax-viewer/);
});

test('loadReferenceContext reads lab.md from the working directory', t => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'gigaam-context-test-'));
    t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
    fs.writeFileSync(path.join(directory, 'lab.md'), '# Lab\n\nReference text.');

    const context = loadReferenceContext(directory);
    assert.equal(context.name, 'lab.md');
    assert.match(context.content, /Reference text/);
});

test('selectReferenceContext keeps the sections relevant to the question', () => {
    const context = {
        content: [
            '## Исходные данные\nКоэффициенты системы.',
            '## Скоростная ошибка\nОшибка зависит от скорости.',
            '## Постоянная времени двигателя\nTдв = 0,9 с и определяется по времени затухания.',
            '## Выводы\nОбщие результаты лабораторной работы.',
        ].join('\n\n'),
    };

    const selected = selectReferenceContext(context, 'Как определена постоянная времени двигателя?', 140);
    assert.match(selected, /Постоянная времени двигателя/);
    assert.match(selected, /0,9 с/);
    assert.doesNotMatch(selected, /Скоростная ошибка/);
});

test('buildCodexArgs forces low-effort GPT-5.6 Sol without agent tools', () => {
    const args = buildCodexArgs({ workingDirectory: '/tmp', codexModel: 'gpt-5.6-sol' });
    const joined = args.join(' ');

    assert.match(joined, /--model gpt-5\.6-sol/);
    assert.match(joined, /model_reasoning_effort="low"/);
    assert.doesNotMatch(joined, /service_tier/);
    assert.match(joined, /features\.shell_tool=false/);
    assert.match(joined, /web_search="disabled"/);
});

test('buildCodexAppServerArgs enables streaming server without agent tools or fast tier', () => {
    const joined = buildCodexAppServerArgs().join(' ');

    assert.match(joined, /^app-server --listen stdio:\/\//);
    assert.match(joined, /model_reasoning_effort="low"/);
    assert.match(joined, /features\.shell_tool=false/);
    assert.match(joined, /web_search="disabled"/);
    assert.match(joined, /mcp_servers=\{\}/);
    assert.doesNotMatch(joined, /service_tier/);
});
