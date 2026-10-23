(function () {
  var root = document.documentElement;
  var themes = ["auto", "light", "dark"];
  var themeLabels = {
    zh: { auto: "主题：跟随系统", light: "主题：浅色", dark: "主题：深色" },
    en: { auto: "Theme: system", light: "Theme: light", dark: "Theme: dark" },
  };
  var themeButton = document.getElementById("theme-toggle");
  var langButton = document.getElementById("lang-toggle");

  function save(key, value) {
    try { localStorage.setItem(key, value); } catch (e) {}
  }

  function updateTitles() {
    var lang = root.dataset.lang === "en" ? "en" : "zh";
    themeButton.title = themeLabels[lang][root.dataset.theme || "auto"];
    langButton.title = lang === "en" ? "切换到中文" : "Switch to English";
  }

  themeButton.addEventListener("click", function () {
    var current = root.dataset.theme || "auto";
    var next = themes[(themes.indexOf(current) + 1) % themes.length];
    if (next === "auto") delete root.dataset.theme;
    else root.dataset.theme = next;
    save("theme", next);
    updateTitles();
  });

  langButton.addEventListener("click", function () {
    var next = root.dataset.lang === "en" ? "zh" : "en";
    save("lang", next);
    root.dataset.lang = next;
    root.lang = next === "en" ? "en" : "zh-CN";
    document.title = document.querySelector("title").dataset[next];
    updateTitles();
  });

  updateTitles();
})();

/* Image lightbox: click a content image to zoom it from its spot to full view. */
(function () {
  var images = document.querySelectorAll(".prose img");
  if (!images.length) return;

  var root = document.documentElement;
  var box = document.createElement("div");
  box.className = "lightbox";
  box.setAttribute("role", "dialog");
  box.setAttribute("aria-modal", "true");
  box.hidden = true;
  box.innerHTML =
    '<button type="button" class="lightbox-close" aria-label="关闭 / Close">' +
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>' +
    "</button><img alt=\"\">";
  document.body.appendChild(box);

  var big = box.querySelector("img");
  var closeButton = box.querySelector(".lightbox-close");
  var source = null;
  var target = null;
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function fit(img) {
    var pad = window.innerWidth < 560 ? 12 : 48;
    var w = img.naturalWidth || img.width;
    var h = img.naturalHeight || img.height;
    var scale = Math.min((window.innerWidth - pad * 2) / w, (window.innerHeight - pad * 2) / h, 1);
    w *= scale;
    h *= scale;
    return { left: (window.innerWidth - w) / 2, top: (window.innerHeight - h) / 2, width: w, height: h };
  }

  function transformTo(rect) {
    return "translate(" + (rect.left - target.left) + "px," + (rect.top - target.top) + "px) " +
      "scale(" + rect.width / target.width + ")";
  }

  function open(img) {
    source = img;
    target = fit(img);
    big.src = img.currentSrc || img.src;
    big.alt = img.alt;
    big.style.left = target.left + "px";
    big.style.top = target.top + "px";
    big.style.width = target.width + "px";
    big.style.height = target.height + "px";
    big.style.transition = "none";
    big.style.transform = transformTo(img.getBoundingClientRect());
    box.hidden = false;
    root.classList.add("lightbox-active");
    img.style.visibility = "hidden";
    void big.offsetWidth;
    big.style.transition = "";
    box.classList.add("is-open");
    big.style.transform = "none";
    closeButton.focus({ preventScroll: true });
  }

  function close() {
    if (!source) return;
    var img = source;
    var rect = img.getBoundingClientRect();
    var onScreen = rect.bottom > 0 && rect.top < window.innerHeight;
    box.classList.remove("is-open");
    if (onScreen) big.style.transform = transformTo(rect);
    else img.style.visibility = "";
    source = null;

    function done() {
      if (source) return;
      box.hidden = true;
      img.style.visibility = "";
      root.classList.remove("lightbox-active");
    }
    if (reduceMotion) done();
    else setTimeout(done, 300);
    img.focus({ preventScroll: true });
  }

  Array.prototype.forEach.call(images, function (img) {
    if (img.closest("a")) return;
    img.classList.add("zoomable");
    img.tabIndex = 0;
    img.addEventListener("click", function () { open(img); });
    img.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(img); }
    });
  });

  box.addEventListener("click", close);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && source) close();
  });
  window.addEventListener("resize", function () {
    if (!source) return;
    target = fit(source);
    big.style.transition = "none";
    big.style.left = target.left + "px";
    big.style.top = target.top + "px";
    big.style.width = target.width + "px";
    big.style.height = target.height + "px";
  });
})();
