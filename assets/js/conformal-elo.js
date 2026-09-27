// A live, explicitly illustrative Bradley–Terry placement against fixed anchors.
function fitSoftElo(probabilities, anchors) {
  let low = Math.min(...anchors) - 10000;
  let high = Math.max(...anchors) + 10000;
  const observedWins = probabilities.reduce((sum, p) => sum + p, 0);
  for (let step = 0; step < 80; step++) {
    const rating = (low + high) / 2;
    const expectedWins = anchors.reduce((sum, anchor) => sum + 1 / (1 + 10 ** ((anchor - rating) / 400)), 0);
    if (expectedWins < observedWins) low = rating;
    else high = rating;
  }
  return (low + high) / 2;
}

function eloConformalQuantile(errors, coverage) {
  const ordered = [...errors].sort((a, b) => a - b);
  return ordered[Math.ceil((ordered.length + 1) * coverage) - 1] ?? Infinity;
}

const eloPipeline = document.getElementById("elo-pipeline");
if (eloPipeline) {
  const anchors = [1000, 1200, 1400];
  // Synthetic calibration errors divided by bootstrap SE. Never used as paper results.
  const errors = [0.12, 0.22, 0.32, 0.45, 0.55, 0.65, 0.8, 0.85, 0.9, 1, 1.05, 1.15, 1.2, 1.35, 1.5, 1.65, 1.8, 2, 2.8];
  const scenarios = { close: [6.3, 4.1, 6.1, 8.1], strong: [8.8, 3.2, 5.2, 7.2] };
  const beta = document.getElementById("elo-beta");
  const interval = document.getElementById("elo-interval-demo");
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const get = (id) => document.getElementById(`elo-${id}`);
  const formatRating = (rating) => Math.round(rating).toLocaleString("en-US");
  const coverage = () => Number(document.querySelector('input[name="elo-coverage"]:checked').value);
  const values = () => [
    ...scenarios[document.querySelector('input[name="elo-scenario"]:checked').value],
    Number(beta.value),
    eloConformalQuantile(errors, coverage()),
  ];
  function paint([score, ...rest]) {
    const [lowScore, middleScore, highScore, scale, q] = rest;
    const referenceScores = [lowScore, middleScore, highScore];
    const probabilities = referenceScores.map((other) => preferenceProbability(score, other, scale));
    probabilities.forEach((p, i) => {
      get(`score-${i}`).textContent = score.toFixed(1);
      get(`ref-${i}`).textContent = referenceScores[i].toFixed(1);
      get(`p-${i}`).textContent = `${(100 * p).toFixed(1)}%`;
      get(`p-bar-${i}`).style.transform = `scaleX(${p})`;
      get(`hard-${i}`).textContent = score > referenceScores[i] ? "Win" : score < referenceScores[i] ? "Loss" : "Tie";
    });
    get("beta-value").textContent = scale.toFixed(2);
    const rating = fitSoftElo(probabilities, anchors);
    get("rating").textContent = formatRating(rating);
    anchors.forEach((anchor, i) => {
      const fitted = 1 / (1 + 10 ** ((anchor - rating) / 400));
      get(`fitted-${i}`).textContent = `${(100 * fitted).toFixed(1)}%`;
    });
    const gap = score - middleScore;
    const targetB = (100 * probabilities[1]).toFixed(1);
    const fittedB = (100 / (1 + 10 ** ((anchors[1] - rating) / 400))).toFixed(1);
    get("score-insight").textContent = `Against B, the score gap is Δs = ${score.toFixed(1)} − ${middleScore.toFixed(1)} = ${gap.toFixed(1)}.`;
    get("prob-insight").textContent = `With β = ${scale.toFixed(2)}, that ${gap.toFixed(1)}-point gap gives a preference probability of ${targetB}%.`;
    get("fit-insight").textContent =
      scale === 0
        ? "At β = 0 there is no preference signal. The three opponent ratings give a fit of 1,200 Elo."
        : `Its ${fittedB}% predicted win rate against B closely matches the ${targetB}% from calibration.`;
    const margin = q * 25;
    const bandStart = 230 - margin * 1.9;
    const bandEnd = 230 + margin * 1.9;
    get("band-fill").setAttribute("x", bandStart);
    get("band-fill").setAttribute("width", bandEnd - bandStart);
    get("band-line").setAttribute("d", `M${bandStart} 94V116 M${bandStart} 105H${bandEnd} M${bandEnd} 94V116`);
    get("band-label").textContent = `${formatRating(rating)} ± ${Math.round(margin)} Elo`;
    get("band-caption").textContent = `${Math.round(coverage() * 100)}% prediction interval · width ${Math.round(margin * 2)} Elo`;
    get("band-desc").textContent =
      `Illustrative interval centered on ${formatRating(rating)} Elo, with margin ${Math.round(margin)} Elo. The bootstrap standard error is fixed at 25 Elo.`;
    get("axis-low").textContent = formatRating(rating - 100);
    get("axis-middle").textContent = formatRating(rating);
    get("axis-high").textContent = formatRating(rating + 100);
    get("error-area").setAttribute("width", q * 130);
    get("q-line").setAttribute("d", `M${35 + q * 130} 42V138`);
    get("quantile").textContent = `${Math.ceil((errors.length + 1) * coverage())}th of ${errors.length} errors → q = ${q.toFixed(1)}`;
  }
  const animation = animateValues(values(), paint, motion);
  function update(event) {
    animation.to(values(), event?.target === beta ? 150 : 700);
  }
  function announce() {
    const [score, a, b, c, scale, q] = values();
    const rating = fitSoftElo(
      [a, b, c].map((other) => preferenceProbability(score, other, scale)),
      anchors
    );
    get("example-status").textContent =
      `Illustrative new-model rating: ${formatRating(rating)} Elo. ${Math.round(coverage() * 100)}% interval: plus or minus ${Math.round(q * 25)} Elo.`;
  }
  eloPipeline.addEventListener("input", update);
  eloPipeline.addEventListener("change", announce);
  interval.addEventListener("change", () => {
    update();
    announce();
  });
  get("reset").addEventListener("click", () => {
    document.querySelector('input[name="elo-scenario"][value="close"]').checked = true;
    document.querySelector('input[name="elo-coverage"][value="0.9"]').checked = true;
    beta.value = "0.5";
    update();
    announce();
  });
  document.querySelectorAll("#elo-pipeline output, #elo-interval-demo output").forEach((output) => output.setAttribute("aria-live", "off"));
  eloPipeline.querySelector(".elo-example-controls").hidden = false;
  eloPipeline.querySelector(".elo-beta-control").hidden = false;
  interval.querySelector(".elo-interval-controls").hidden = false;
}
