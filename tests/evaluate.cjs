// Evaluate library calls through the JS engine's normal module loader.
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const enginePath = path.resolve(process.env.FLATPPL_JS_DIR || path.join(root, '../flatppl-js'), 'packages/engine');
const engine = require(path.join(enginePath, 'index.ts'));
const { createWorkerHandler } = require(path.join(enginePath, 'worker.ts'));

(async () => {
  const request = JSON.parse(fs.readFileSync(0, 'utf8'));
  const expressions = Array.isArray(request) ? request : null;
  const modulePaths = ['main.flatppl', ...['src', 'examples'].flatMap(dir =>
    fs.readdirSync(path.join(root, dir), { recursive: true })
      .filter(name => name.endsWith('.flatppl')).map(name => dir + '/' + name))];
  const sources = Object.fromEntries(modulePaths
    .map(name => [path.join(root, name), fs.readFileSync(path.join(root, name), 'utf8')]));
  const source = expressions ? 'flatppl_compat = "0.1"\na = load_module("../main.flatppl")\n'
    + expressions.map((expression, i) => `probe${i} = ${expression}`).join('\n') : request.source;
  const model = engine.processSource(source, {
    path: path.join(__dirname, 'query.flatppl'), bundle: { sources },
  });
  const errors = model.diagnostics.filter(d => d.severity === 'error');
  if (errors.length) throw new Error(JSON.stringify(errors));
  const result = engine.orchestrator.buildDerivations(model.linkedBindings, {
    moduleRegistry: model.linkedModuleRegistry,
  });
  const names = expressions ? expressions.map((_, i) => `probe${i}`) : request.bindings;
  const worker = createWorkerHandler();
  const seed = 47710;
  worker.handle({ type: 'init', seed });
  const ctx = {
    ...result, sampleCount: request.sampleCount || 1, rootKey: seed, rootSeed: seed,
    marginalizationCount: 64,
    sendWorker: async message => {
      const reply = await worker.handle(message);
      if (reply?.type === 'error') throw new Error(JSON.stringify(reply));
      return reply;
    },
    getMeasure: name => engine.materialiser.materialiseMeasure(name, ctx),
  };
  const values = await Promise.all(names.map(async name => {
    if (result.fixedValues.has(name)) return result.fixedValues.get(name);
    const measure = await ctx.getMeasure(name);
    return request.sampleCount ? measure : measure.samples[0];
  }));
  process.stdout.write(JSON.stringify(values, (_key, value) =>
    typeof value === 'number' && !Number.isFinite(value) ? String(value) : value));
})().catch(error => { console.error(error); process.exitCode = 1; });
