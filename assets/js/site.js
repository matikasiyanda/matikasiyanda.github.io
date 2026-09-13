(function () {
  var root = document.documentElement;

  // ---------- matrix rain ----------
  var canvas = document.getElementById("rain");
  var ctx = canvas.getContext("2d");
  var glyphs = "01アイウエオカキクケコサシスセソタチツテトナニヌネノ0123456789ABCDEF<>/{}$#";
  var size = 16, columns = [], timer = null;

  function resize() {
    var dpr = window.devicePixelRatio || 1;
    canvas.width = innerWidth * dpr;
    canvas.height = innerHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var count = Math.ceil(innerWidth / size);
    columns = Array.from({ length: count }, function (_, i) {
      return columns[i] !== undefined ? columns[i] : Math.random() * innerHeight / size;
    });
  }

  function frame() {
    ctx.fillStyle = "rgba(0, 0, 0, 0.08)";
    ctx.fillRect(0, 0, innerWidth, innerHeight);
    ctx.font = size + "px Menlo, Monaco, monospace";
    for (var i = 0; i < columns.length; i++) {
      var y = columns[i] * size;
      var ch = glyphs[(Math.random() * glyphs.length) | 0];
      ctx.fillStyle = Math.random() < 0.04 ? "#d6ffd1" : "#28fe14";
      ctx.fillText(ch, i * size, y);
      if (y > innerHeight && Math.random() > 0.975) columns[i] = 0;
      else columns[i] += 1;
    }
  }

  function start() { if (!timer) { resize(); timer = setInterval(frame, 55); } }
  function stop() { clearInterval(timer); timer = null; ctx.clearRect(0, 0, innerWidth, innerHeight); }

  var btn = document.getElementById("rain-toggle");
  function sync() {
    var on = root.dataset.rain === "on";
    btn.textContent = on ? "rain: on" : "rain: off";
    btn.setAttribute("aria-pressed", on);
    on ? start() : stop();
  }
  btn.addEventListener("click", function () {
    root.dataset.rain = root.dataset.rain === "on" ? "off" : "on";
    localStorage.setItem("rain", root.dataset.rain);
    sync();
  });
  addEventListener("resize", function () { if (timer) resize(); });
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) { clearInterval(timer); timer = null; }
    else if (root.dataset.rain === "on") start();
  });
  sync();

  // ---------- math (kramdown emits \( \) and \[ \]) ----------
  function renderMath() {
    if (window.renderMathInElement) {
      renderMathInElement(document.querySelector("main"), {
        delimiters: [
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false }
        ],
        throwOnError: false
      });
    }
  }
  if (document.readyState === "complete") renderMath();
  else addEventListener("load", renderMath);

  // ---------- open figures at full size ----------
  document.querySelectorAll(".prose img").forEach(function (img) {
    img.loading = "lazy";
    img.addEventListener("click", function () { window.open(img.src, "_blank", "noopener"); });
  });
})();
