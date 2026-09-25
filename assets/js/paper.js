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

const scoreGap = document.getElementById("score-gap");
if (scoreGap) {
  scoreGap.disabled = false;
  scoreGap.addEventListener("input", () => {
    const gap = Number(scoreGap.value);
    document.getElementById("gap-value").textContent = gap.toFixed(1);
    document.getElementById("score-a").textContent = (5 + gap / 2).toFixed(1);
    document.getElementById("score-b").textContent = (5 - gap / 2).toFixed(1);
    document.getElementById("win-probability").value = `${(100 / (1 + Math.exp(-0.5 * gap))).toFixed(1)}%`;
  });
}
