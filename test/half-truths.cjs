// Run: node test/half-truths.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const read = (file) => fs.readFileSync(path.join(__dirname, "..", file), "utf8");
const html = read("_pages/half-truths.html");
const csv = read("assets/img/papers/half-truths/results.csv")
  .trim()
  .split(/\r?\n/)
  .slice(1)
  .map((line) => line.split(","));
const cells = [...html.matchAll(/<td data-values="([^"]+)" data-best="([^"]+)"/g)].map((m) => {
  const score = {
    textContent: "",
    style: {
      setProperty(k, v) {
        this[k] = v;
      },
    },
    classList: {
      toggle(k, v) {
        this[k] = v;
      },
    },
  };
  return { dataset: { values: m[1], best: m[2] }, score, querySelector: () => score };
});
const bars = [...html.matchAll(/class="result-bar-row[^\"]*"\s+data-dataset="([^"]+)"\s+data-model="([^"]+)"\s+data-values="([^"]+)"/g)].map((m) => {
  const bar = { style: {} },
    output = {};
  return { dataset: { dataset: m[1], model: m[2], values: m[3] }, bar, output, querySelector: (s) => (s === ".result-bar" ? bar : output) };
});
assert.equal(cells.length, 48);
assert.equal(bars.length, 9);
const radio = [0, 1, 2].map((n) => ({ value: String(n), checked: n === 0 }));
const metric = {
  hidden: true,
  addEventListener: (event, fn) => {
    metric.change = fn;
  },
  querySelector: () => radio.find((r) => r.checked),
  querySelectorAll: () => radio,
};
const control = { hidden: true };
const select = {
  value: "0",
  closest: () => control,
  addEventListener: (event, fn) => {
    select.change = fn;
  },
};
const caption = {};
const ctx = {
  document: {
    getElementById: (id) => ({ "half-table-metric": select, "half-table-caption": caption })[id] || null,
    querySelector: (s) => (s === ".metric-switch" ? metric : null),
    querySelectorAll: (s) => (s === ".result-bar-row" ? bars : cells),
  },
};
vm.createContext(ctx);
vm.runInContext(read("assets/js/paper.js"), ctx);
assert.equal(metric.hidden, false);
assert.equal(control.hidden, false);
for (const source of ["chart", "table"]) {
  for (const index of [0, 1, 2, 0]) {
    if (source === "table") {
      select.value = String(index);
      select.change();
    } else {
      radio.forEach((r) => {
        r.checked = Number(r.value) === index;
      });
      metric.change();
    }
    assert.equal(select.value, String(index));
    assert.equal(radio.find((r) => r.checked).value, String(index));
    assert.equal(caption.textContent, `${["Overall", "Entity", "Relation"][index]} accuracy (%)`);
    cells.forEach((cell, i) => {
      const value = Number(csv[i][3 + index]);
      assert.equal(cell.score.textContent, value.toFixed(1));
      assert.equal(parseFloat(cell.score.style["--score"]), Number(value.toFixed(1)));
      const max = Math.max(...csv.filter((r) => r[0] === csv[i][0]).map((r) => Number(Number(r[3 + index]).toFixed(1))));
      assert.equal(cell.score.classList["is-best"], Number(value.toFixed(1)) === max);
    });
    bars.forEach((row) => {
      const record = csv.find((r) => r[0] === row.dataset.dataset && r[1] === row.dataset.model);
      assert.equal(row.output.textContent, Number(record[3 + index]).toFixed(1));
      assert.equal(parseFloat(row.bar.style.width), Number(record[3 + index]));
    });
  }
}
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
assert.equal(ids.length, new Set(ids).size);
assert(!html.includes("<details"));
assert(!html.includes('class="resource-strip"'));
assert(html.indexOf('id="idea"') < html.indexOf('id="abstract"'));
assert(html.indexOf('id="method"') < html.indexOf('id="results"'));
assert(html.indexOf('id="results"') < html.indexOf('id="takeaway"'));
console.log("Half-Truths: both control paths, all 144 table values, best-score highlights, chart synchronization and section order passed.");
