const copyButton = document.getElementById("copy-citation");
if (copyButton && navigator.clipboard) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    const status = document.getElementById("copy-status");
    try {
      await navigator.clipboard.writeText(document.getElementById("bibtex").textContent.trim());
      status.textContent = "Citation copied to clipboard.";
      copyButton.textContent = "Copied!";
    } catch {
      status.textContent = "Could not copy. Select and copy the citation below.";
      copyButton.textContent = "Select the text below";
    }
  });
}

function preferenceProbability(scoreA, scoreB, beta) {
  return 1 / (1 + Math.exp(-beta * (scoreA - scoreB)));
}

// Retarget from the last painted frame so rapid clicks never queue or jump.
function animateValues(initial, paint, motion) {
  let current = initial.slice();
  let target = current.slice();
  let frame;
  function finish() {
    cancelAnimationFrame(frame);
    current = target.slice();
    paint(current);
  }
  motion.addEventListener("change", () => {
    if (motion.matches) finish();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) finish();
  });
  paint(current);
  return {
    to(next, duration = 1000) {
      cancelAnimationFrame(frame);
      target = next.slice();
      if (motion.matches || duration === 0) {
        finish();
        return;
      }
      const from = current.slice();
      const start = performance.now();
      function tick(now) {
        const t = Math.min(1, Math.max(0, (now - start) / duration));
        const eased = t * t * (3 - 2 * t);
        current = t === 1 ? target.slice() : from.map((value, i) => value + (target[i] - value) * eased);
        paint(current);
        if (t < 1) frame = requestAnimationFrame(tick);
      }
      frame = requestAnimationFrame(tick);
    },
  };
}

const halfDemo = document.getElementById("half-demo");
if (halfDemo) {
  const examples = JSON.parse(document.getElementById("half-example-data").textContent);
  const selector = document.getElementById("half-example");
  const choices = [...halfDemo.querySelectorAll('input[name="half-caption"]')];
  const pause = document.getElementById("half-pause");
  const photo = document.getElementById("half-image");
  const imageBase = photo.getAttribute("src").replace(/[^/]+$/, "");
  const caption = document.getElementById("half-caption-text");
  const addition = document.getElementById("half-caption-addition");
  const explanation = document.getElementById("half-explanation");
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const bars = ["clip", "cs"].map((id) => ({
    bar: document.getElementById(`half-${id}-bar`),
    score: document.getElementById(`half-${id}-score`),
    anchor: document.getElementById(`half-${id}-anchor`),
  }));
  const models = ["CLIP", "CS-CLIP"];
  const transition = animateValues(
    models.map((name) => examples[0].scores[name].s_anchor),
    (values) =>
      bars.forEach(({ bar, score }, i) => {
        bar.style.transform = `scaleX(${values[i] / 0.5})`;
        score.textContent = values[i].toFixed(3);
      }),
    motion
  );
  let timer;
  let paused = false;
  let visible = false;
  function render(duration = 1000) {
    const example = examples[Number(selector.value)];
    const kind = choices.find((choice) => choice.checked).value;
    photo.src = imageBase + example.image;
    photo.alt = example.alt;
    document.getElementById("half-caption-anchor").textContent = example.anchor;
    addition.textContent = example[kind].slice(example.anchor.length);
    caption.dataset.kind = kind;
    bars.forEach(({ anchor }, i) => {
      anchor.style.left = `${(100 * example.scores[models[i]].s_anchor) / 0.5}%`;
    });
    transition.to(
      models.map((name) => example.scores[name][`s_${kind}`]),
      duration
    );
    explanation.textContent =
      kind === "anchor"
        ? "Start with a correct description. The dashed lines mark its score before adding a detail."
        : kind === "full"
          ? "The added detail is correct. Both models score this caption above the original description."
          : `CLIP rewards the false detail (+${(-example.scores.CLIP.delta).toFixed(3)}). CS-CLIP lowers its score (−${example.scores["CS-CLIP"].delta.toFixed(3)}).`;
  }
  function schedule() {
    clearTimeout(timer);
    pause.hidden = motion.matches;
    pause.textContent = paused ? "Resume animation" : "Pause animation";
    pause.setAttribute("aria-pressed", String(paused));
    explanation.setAttribute("aria-live", paused || motion.matches ? "polite" : "off");
    if (paused || motion.matches || !visible || document.hidden) return;
    timer = setTimeout(() => {
      const index = choices.findIndex((choice) => choice.checked);
      choices[(index + 1) % choices.length].checked = true;
      render();
      schedule();
    }, 4000);
  }
  choices.forEach((choice) =>
    choice.addEventListener("change", () => {
      paused = true;
      render();
      schedule();
    })
  );
  selector.addEventListener("change", () => {
    choices[0].checked = true;
    render(0);
    schedule();
  });
  pause.addEventListener("click", () => {
    paused = !paused;
    schedule();
  });
  motion.addEventListener("change", schedule);
  document.addEventListener("visibilitychange", schedule);
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    schedule();
  }).observe(halfDemo);
  halfDemo.querySelector(".demo-controls").hidden = false;
  halfDemo.querySelector(".demo-choices").hidden = false;
  render(0);
}

const eloDemo = document.getElementById("elo-demo");
if (eloDemo) {
  const scoreA = document.getElementById("demo-score-a");
  const scoreB = document.getElementById("demo-score-b");
  const beta = document.getElementById("demo-beta");
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const elements = Object.fromEntries(
    ["score-a-value", "score-b-value", "beta-value", "hard", "soft", "probability-curve", "probability-point", "point-guide"].map((id) => [
      id,
      document.getElementById(`demo-${id}`),
    ])
  );
  let animateControls = false;
  function paintPreference([a, b, scale]) {
    const gap = a - b;
    const probability = preferenceProbability(a, b, scale);
    if (animateControls) {
      scoreA.value = a;
      scoreB.value = b;
      beta.value = scale;
    }
    elements["score-a-value"].textContent = a.toFixed(1);
    elements["score-b-value"].textContent = b.toFixed(1);
    elements["beta-value"].textContent = scale.toFixed(2);
    // Hard targets deliberately remain discrete, even during the animation.
    elements.hard.textContent = `${gap > 0 ? 100 : gap < 0 ? 0 : 50}%`;
    elements.soft.textContent = `${(100 * probability).toFixed(1)}%`;
    document.getElementById("demo-hard-bar").style.transform = `scaleX(${gap > 0 ? 1 : gap < 0 ? 0 : 0.5})`;
    document.getElementById("demo-soft-bar").style.transform = `scaleX(${probability})`;
    const x = 50 + ((gap + 9) / 18) * 480;
    const y = 215 - probability * 190;
    elements["probability-curve"].setAttribute(
      "d",
      Array.from({ length: 181 }, (_, i) => {
        const difference = -9 + i / 10;
        return `${i ? "L" : "M"}${(50 + (i / 180) * 480).toFixed(2)},${(215 - preferenceProbability(difference, 0, scale) * 190).toFixed(2)}`;
      }).join(" ")
    );
    elements["probability-point"].setAttribute("cx", x);
    elements["probability-point"].setAttribute("cy", y);
    elements["point-guide"].setAttribute("d", `M50 ${y}H${x}V215`);
  }
  const transition = animateValues([6.3, 6.1, 0.5], paintPreference, motion);
  function setPreference(values, duration, moveControls) {
    animateControls = moveControls;
    transition.to(values, duration);
    const [a, b, scale] = values;
    document.getElementById("elo-demo-explanation").textContent =
      scale === 0
        ? "At β = 0, every score gap maps to 50%: the mapping carries no preference signal."
        : a === b
          ? "Equal scores give a 50% target for both methods, whatever β is."
          : "The soft target changes continuously with the score gap. The hard target stays at a win or loss until the winner changes.";
  }
  eloDemo.addEventListener("input", () => setPreference([Number(scoreA.value), Number(scoreB.value), Number(beta.value)], 120, false));
  eloDemo.querySelectorAll("[data-scores]").forEach((button) =>
    button.addEventListener("click", () => {
      setPreference([...button.dataset.scores.split(",").map(Number), Number(beta.value)], 1000, true);
    })
  );
  document.getElementById("demo-swap").addEventListener("click", () => {
    setPreference([Number(scoreB.value), Number(scoreA.value), Number(beta.value)], 1000, true);
  });
  document.getElementById("demo-reset").addEventListener("click", () => setPreference([6.3, 6.1, 0.5], 1000, true));
  setPreference([6.3, 6.1, 0.5], 0, false);
  eloDemo.hidden = false;
}

const scatterDataset = document.getElementById("scatter-dataset");
if (scatterDataset) {
  const plot = document.getElementById("compositional-scatter");
  const download = document.getElementById("scatter-download");
  const originalSource = plot.getAttribute("src");
  scatterDataset.closest(".scatter-controls").hidden = false;
  scatterDataset.addEventListener("change", () => {
    plot.src = originalSource.replace("compositional-coco", `compositional-${scatterDataset.value}`);
    download.href = plot.src.replace(/\.svg$/, ".pdf");
    plot.alt = `Sixteen models: compositional I2T accuracy versus Half-Truth accuracy on ${scatterDataset.selectedOptions[0].textContent}.`;
  });
}

const metricSwitch = document.querySelector(".metric-switch");
const tableMetric = document.getElementById("half-table-metric");
function updateHalfResults(index) {
  document.querySelectorAll(".result-bar-row").forEach((row) => {
    const value = JSON.parse(row.dataset.values)[index];
    row.querySelector(".result-bar").style.width = `${value}%`;
    row.querySelector("output").textContent = value.toFixed(1);
  });
  document.querySelectorAll("#half-results td[data-values]").forEach((cell) => {
    const value = JSON.parse(cell.dataset.values)[index];
    const score = cell.querySelector(".ht-table-score");
    score.textContent = value.toFixed(1);
    score.style.setProperty("--score", `${value}%`);
    score.classList.toggle("is-best", JSON.parse(cell.dataset.best)[index]);
  });
  if (tableMetric) {
    tableMetric.value = String(index);
    document.getElementById("half-table-caption").textContent = `${["Overall", "Entity", "Relation"][index]} accuracy (%)`;
  }
  if (metricSwitch) {
    metricSwitch.querySelectorAll("input").forEach((input) => {
      input.checked = Number(input.value) === index;
    });
  }
}
if (metricSwitch) {
  metricSwitch.hidden = false;
  metricSwitch.addEventListener("change", () => updateHalfResults(Number(metricSwitch.querySelector("input:checked").value)));
}
if (tableMetric) {
  tableMetric.closest(".ht-table-control").hidden = false;
  tableMetric.addEventListener("change", () => updateHalfResults(Number(tableMetric.value)));
}
