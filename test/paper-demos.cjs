// Run without a browser: node test/paper-demos.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const context = { document: { getElementById: () => null, querySelector: () => null, querySelectorAll: () => [] } };
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, "../assets/js/paper.js"), "utf8"), context);
const p = context.preferenceProbability;
assert.equal(p(5, 5, 2), 0.5);
assert.equal(p(10, 1, 0), 0.5);
assert.equal((100 * p(6.3, 6.1, 0.5)).toFixed(1), "52.5");
assert.equal((100 * p(8.8, 3.2, 0.5)).toFixed(1), "94.3");
for (const beta of [0, 0.05, 0.5, 1, 2]) {
  for (let gap = -9; gap <= 9; gap += 0.1) {
    assert(p(gap, 0, beta) > 0 && p(gap, 0, beta) < 1);
    assert(Math.abs(p(gap, 0, beta) + p(-gap, 0, beta) - 1) < 1e-12);
    assert(p(gap + 0.1, 0, beta) >= p(gap, 0, beta));
  }
}
assert(p(6.3, 6.1, 2) > p(6.3, 6.1, 0.5));
console.log("Preference mapping: ties, zero beta, presets, symmetry, range, and monotonicity passed.");

// Exercise the actual animation clock without requiring Chromium.
let now = 0,
  id = 0;
const frames = new Map();
let rendered;
const events = {};
context.performance = { now: () => now };
context.requestAnimationFrame = (callback) => {
  frames.set(++id, callback);
  return id;
};
context.cancelAnimationFrame = (key) => frames.delete(key);
context.document.addEventListener = (name, callback) => {
  events[name] = callback;
};
const motion = {
  matches: false,
  addEventListener: (_, callback) => {
    events.motion = callback;
  },
};
const animation = context.animateValues(
  [0, 10],
  (values) => {
    rendered = [...values];
  },
  motion
);
function tick(time) {
  now = time;
  const pending = [...frames.values()];
  frames.clear();
  pending.forEach((callback) => callback(time));
}
assert.deepEqual(rendered, [0, 10]);
animation.to([10, 0], 1000);
tick(250);
assert.deepEqual(rendered, [1.5625, 8.4375]);
tick(500);
assert.deepEqual(rendered, [5, 5]);
animation.to([20, 15], 1000);
assert.equal(frames.size, 1);
tick(500);
assert.deepEqual(rendered, [5, 5]);
tick(1000);
assert.deepEqual(rendered, [12.5, 10]);
tick(1500);
assert.deepEqual(rendered, [20, 15]);
assert.equal(frames.size, 0);
animation.to([30, 25], 1000);
motion.matches = true;
events.motion();
assert.deepEqual(rendered, [30, 25]);
assert.equal(frames.size, 0);
animation.to([2, 4]);
assert.deepEqual(rendered, [2, 4]);
motion.matches = false;
animation.to([6, 8]);
context.document.hidden = true;
events.visibilitychange();
assert.deepEqual(rendered, [6, 8]);
assert.equal(frames.size, 0);
console.log("Animation timing: interpolation, interrupted transitions, exact endpoints, reduced motion, and background cleanup passed.");
