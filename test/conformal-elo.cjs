// Run: node test/conformal-elo.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ctx = { document: { getElementById: () => null, querySelector: () => null } };
vm.createContext(ctx);
for (const file of ["paper.js", "conformal-elo.js"]) vm.runInContext(fs.readFileSync(path.join(__dirname, "../assets/js", file), "utf8"), ctx);
const anchors = [1000, 1200, 1400];
// A fit must recover a known rating from its exact Bradley–Terry probabilities.
for (const expected of [700, 1000, 1200, 1500, 2200]) {
  const probabilities = anchors.map((anchor) => 1 / (1 + 10 ** ((anchor - expected) / 400)));
  assert(Math.abs(ctx.fitSoftElo(probabilities, anchors) - expected) < 1e-6);
}
const close = [4.1, 6.1, 8.1].map((score) => ctx.preferenceProbability(6.3, score, 0.5));
const strong = [3.2, 5.2, 7.2].map((score) => ctx.preferenceProbability(8.8, score, 0.5));
assert.equal(Math.round(ctx.fitSoftElo(close, anchors)), 1218);
assert.equal(Math.round(ctx.fitSoftElo(strong, anchors)), 1524);
assert.equal(Math.round(ctx.fitSoftElo([0.5, 0.5, 0.5], anchors)), 1200);
// The computed rating must maximize the stated soft Bradley–Terry likelihood.
for (const targets of [close, strong]) {
  const rating = ctx.fitSoftElo(targets, anchors);
  const logLikelihood = (r) =>
    targets.reduce((sum, p, i) => {
      const q = 1 / (1 + 10 ** ((anchors[i] - r) / 400));
      return sum + p * Math.log(q) + (1 - p) * Math.log1p(-q);
    }, 0);
  for (const offset of [-100, -10, -1, 1, 10, 100]) assert(logLikelihood(rating) > logLikelihood(rating + offset));
}
// Finite-sample quantile uses ceil((n+1)*coverage), not an interpolated percentile.
const errors = [0.12, 0.22, 0.32, 0.45, 0.55, 0.65, 0.8, 0.85, 0.9, 1, 1.05, 1.15, 1.2, 1.35, 1.5, 1.65, 1.8, 2, 2.8];
assert.equal(ctx.eloConformalQuantile(errors, 0.9), 2);
assert.equal(ctx.eloConformalQuantile(errors, 0.95), 2.8);
assert.equal(ctx.eloConformalQuantile([0.1, 0.2], 0.95), Infinity);
assert.deepEqual(errors, [0.12, 0.22, 0.32, 0.45, 0.55, 0.65, 0.8, 0.85, 0.9, 1, 1.05, 1.15, 1.2, 1.35, 1.5, 1.65, 1.8, 2, 2.8]);
const source = fs.readFileSync(path.join(__dirname, "../_pages/conformal-elo.html"), "utf8");
const ids = [...source.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
assert.equal(ids.length, new Set(ids).size);
assert(source.indexOf('id="problem"') < source.indexOf('id="abstract"'));
assert(source.indexOf('id="abstract"') < source.indexOf('id="score-gap"'));
assert(source.indexOf('id="score-gap"') < source.indexOf('id="method"'));
assert(source.indexOf('id="method"') < source.indexOf('id="intervals"'));
assert(source.indexOf('id="intervals"') < source.indexOf('id="results"'));
assert(!source.includes("<details"));
assert(source.includes("layout: half-truths"));
assert(source.includes("Explore the calculation"));
assert(source.includes("Tables 2–3"));
console.log("Known-rating recovery, worked examples, conformal quantiles, section order, shared layout and visible content passed.");

// Exercise the actual page event handlers with the source page's elements.
const nodes = new Map(
  ids.map((id) => [
    id,
    {
      hidden: false,
      style: {},
      attributes: {},
      events: {},
      value: "",
      checked: false,
      setAttribute(name, value) {
        this.attributes[name] = String(value);
      },
      addEventListener(name, fn) {
        this.events[name] = fn;
      },
      querySelector(selector) {
        return nodes.get(selector);
      },
    },
  ])
);
for (const selector of [".elo-example-controls", ".elo-beta-control", ".elo-interval-controls"]) nodes.set(selector, { hidden: true });
const radios = {
  "elo-scenario": [
    { value: "close", checked: true },
    { value: "strong", checked: false },
  ],
  "elo-coverage": [
    { value: "0.9", checked: true },
    { value: "0.95", checked: false },
  ],
};
const doc = {
  getElementById: (id) => nodes.get(id) || null,
  querySelector(selector) {
    const name = /name="([^"]+)"/.exec(selector)?.[1];
    if (!name) return null;
    const value = /value="([^"]+)"/.exec(selector)?.[1];
    return radios[name].find((r) => (value ? r.value === value : r.checked));
  },
  querySelectorAll: () => [],
  addEventListener: () => {},
};
nodes.get("elo-beta").value = ".5";
const ui = {
  document: doc,
  window: { matchMedia: () => ({ matches: true, addEventListener: () => {} }) },
  performance: { now: () => 0 },
  cancelAnimationFrame: () => {},
  requestAnimationFrame: () => {
    throw Error("Reduced motion should not schedule frames");
  },
};
vm.createContext(ui);
for (const file of ["paper.js", "conformal-elo.js"]) vm.runInContext(fs.readFileSync(path.join(__dirname, "../assets/js", file), "utf8"), ui);
assert.equal(nodes.get("elo-rating").textContent, "1,218");
assert.equal(nodes.get("elo-p-1").textContent, "52.5%");
assert.equal(nodes.get("elo-fitted-0").textContent, "77.8%");
assert.equal(nodes.get("elo-fitted-1").textContent, "52.6%");
assert.equal(nodes.get("elo-fitted-2").textContent, "26.0%");
assert.match(nodes.get("elo-prob-insight").textContent, /0.2-point gap gives a preference probability of 52.5%/);
assert.match(nodes.get("elo-fit-insight").textContent, /52.6%.*52.5%/);
assert.equal(nodes.get(".elo-example-controls").hidden, false);
radios["elo-scenario"][0].checked = false;
radios["elo-scenario"][1].checked = true;
nodes.get("elo-pipeline").events.input({ target: radios["elo-scenario"][1] });
assert.equal(nodes.get("elo-rating").textContent, "1,524");
assert.equal(nodes.get("elo-band-label").textContent, "1,524 ± 50 Elo");
assert.equal(nodes.get("elo-fitted-0").textContent, "95.3%");
assert.equal(nodes.get("elo-fitted-2").textContent, "67.1%");
assert.match(nodes.get("elo-prob-insight").textContent, /3.6-point gap gives a preference probability of 85.8%/);
assert.match(nodes.get("elo-fit-insight").textContent, /86.6%.*85.8%/);
radios["elo-coverage"][0].checked = false;
radios["elo-coverage"][1].checked = true;
nodes.get("elo-interval-demo").events.change();
assert.equal(nodes.get("elo-band-label").textContent, "1,524 ± 70 Elo");
assert.equal(nodes.get("elo-quantile").textContent, "19th of 19 errors → q = 2.8");
nodes.get("elo-beta").value = "0";
nodes.get("elo-pipeline").events.input({ target: nodes.get("elo-beta") });
assert.equal(nodes.get("elo-rating").textContent, "1,200");
assert.match(nodes.get("elo-fit-insight").textContent, /no preference signal/);
// Simulate native radio-group exclusivity during reset.
radios["elo-scenario"][1].checked = false;
radios["elo-coverage"][1].checked = false;
nodes.get("elo-reset").events.click();
assert.equal(nodes.get("elo-rating").textContent, "1,218");
assert.equal(nodes.get("elo-band-label").textContent, "1,218 ± 50 Elo");
console.log("Actual UI handlers: score presets, calibration strength, coverage, reset and reduced motion passed.");
