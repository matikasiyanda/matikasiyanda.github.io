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

  // in-page image viewer: enlarge on click, close with Esc, the button or a click outside
  var viewer = document.createElement("div");
  viewer.className = "lightbox";
  viewer.setAttribute("role", "dialog");
  viewer.setAttribute("aria-modal", "true");
  viewer.hidden = true;
  viewer.innerHTML = '<button type="button" class="lightbox-close" aria-label="Close image">&times;</button>' +
    '<figure class="lightbox-figure"><img alt=""><figcaption></figcaption></figure>';
  document.body.appendChild(viewer);
  var viewerImg = viewer.querySelector("img");
  var viewerCap = viewer.querySelector("figcaption");
  var lastFocus = null;

  function openViewer(img) {
    lastFocus = document.activeElement;
    viewerImg.src = img.currentSrc || img.src;
    viewerImg.alt = img.alt;
    viewerImg.classList.toggle("no-invert", img.classList.contains("no-invert"));
    viewerCap.textContent = img.alt;
    viewer.hidden = false;
    document.documentElement.classList.add("lightbox-open");
    viewer.querySelector(".lightbox-close").focus();
  }
  function closeViewer() {
    viewer.hidden = true;
    viewerImg.removeAttribute("src");
    document.documentElement.classList.remove("lightbox-open");
    if (lastFocus) lastFocus.focus();
  }
  viewer.addEventListener("click", function (e) { if (e.target !== viewerImg) closeViewer(); });
  addEventListener("keydown", function (e) { if (e.key === "Escape" && !viewer.hidden) closeViewer(); });

  document.querySelectorAll(".prose img").forEach(function (img) {
    img.loading = "lazy";
    img.tabIndex = 0;
    img.setAttribute("role", "button");
    img.addEventListener("click", function () { openViewer(img); });
    img.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openViewer(img); }
    });
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
