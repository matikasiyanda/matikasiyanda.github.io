(function () {
  // kramdown emits \( \) and \[ \]
  function renderMath() {
    if (!window.renderMathInElement) return;
    renderMathInElement(document.querySelector("main"), {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false
    });
  }
  if (document.readyState === "complete") renderMath();
  else addEventListener("load", renderMath);

  document.querySelectorAll(".prose img").forEach(function (img) {
    img.loading = "lazy";
    img.addEventListener("click", function () { window.open(img.src, "_blank", "noopener"); });
  });
})();
