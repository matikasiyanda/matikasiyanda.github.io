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

// copy buttons for BibTeX blocks
document.querySelectorAll(".copy-btn").forEach(function (btn) {
  btn.addEventListener("click", function () {
    var target = document.getElementById(btn.dataset.copyTarget);
    navigator.clipboard.writeText(target.innerText).then(function () {
      btn.textContent = "Copied";
      setTimeout(function () { btn.textContent = "Copy"; }, 1500);
    });
  });
});

// share: copy link + native share sheet where the browser supports it
document.querySelectorAll(".copy-link").forEach(function (btn) {
  btn.addEventListener("click", function () {
    navigator.clipboard.writeText(btn.dataset.url).then(function () {
      var old = btn.textContent; btn.textContent = "Copied";
      setTimeout(function () { btn.textContent = old; }, 1500);
    });
  });
});
if (navigator.share) {
  document.querySelectorAll(".native-share").forEach(function (btn) {
    btn.hidden = false;
    btn.addEventListener("click", function () {
      navigator.share({ title: btn.dataset.title, url: btn.dataset.url }).catch(function () {});
    });
  });
}

// jumping to an embedded section (e.g. #glossary) opens its collapsed block
function openTargetDetails() {
  var id = decodeURIComponent(location.hash.slice(1));
  var el = id && document.getElementById(id);
  if (!el) return;
  var det = el.tagName === "DETAILS" ? el : el.nextElementSibling;
  if (det && det.tagName === "DETAILS") det.open = true;
}
addEventListener("hashchange", openTargetDetails);
openTargetDetails();
