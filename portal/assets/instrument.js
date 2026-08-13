/* Load warehouse FIGURE-INDEX / summaries when served from _site/data/. */
(function () {
  const status = document.querySelector("[data-index-status]");
  const url = "data/figures.json";

  function mark(ok, detail) {
    if (!status) return;
    status.textContent = ok ? detail : "summary missing — see papers/figs/PIPELINE.md";
  }

  fetch(url)
    .then(function (res) {
      if (!res.ok) throw new Error("missing " + url);
      return res.json();
    })
    .then(function (index) {
      if (!index || index.paper_id !== "A") throw new Error("unexpected index");
      mark(true, "papers/FIGURE-INDEX.json · " + (index.figures || []).length + " figures");
    })
    .catch(function () {
      mark(false);
    });
})();
