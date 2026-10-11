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
