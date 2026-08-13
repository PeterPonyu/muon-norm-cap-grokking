(function loadFigureIndex() {
  const preview = document.getElementById("index-preview");
  const sources = ["data/figures.json", "../papers/FIGURE-INDEX.json"];

  function render(index) {
    if (!preview) {
      return;
    }
    const ids = (index.figures || []).map(function (figure) {
      return figure.id;
    });
    preview.textContent = ids.join("\n");
  }

  function tryFetch(index) {
    if (index >= sources.length) {
      return;
    }
    fetch(sources[index])
      .then(function (response) {
        if (!response.ok) {
          throw new Error("missing index");
        }
        return response.json();
      })
      .then(render)
      .catch(function () {
        tryFetch(index + 1);
      });
  }

  tryFetch(0);
})();
