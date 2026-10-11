# RoschildRui.github.io

基于 [Jekyll](https://github.com/jekyll/jekyll) 的极简个人主页，推送到 `master` 后由 GitHub Actions 自动构建发布。

首页只有三块：**关于**、**文章**、**想法**。

## 目录

```
_includes/intro.md     首页的自我介绍（中文）
_includes/intro_en.md  首页的自我介绍（英文）
about.md / about_en.md 完整介绍（中 / 英）
_posts/                文章（Markdown 或 HTML，英文版为同名 + _en）
_thoughts/             想法（每条一个文件）
_plugins/              把 xxx_en.md 和 xxx.md 配成一对的小插件
img/                   图片
assets/css/style.css   全部样式
```

## 写文章

文件放进 `_posts/`，文件名格式为 `YYYY-MM-DD-标题.md`，日期决定排序和链接（`/YYYY/MM/DD/标题/`）。

**Markdown 文章**：

```markdown
---
title: 文章标题
subtitle: 可选的副标题
math: true          # 可选，需要渲染 $公式$ 时打开
---

正文……
```

**HTML 片段**（只有正文，套用站点样式）：同上，把扩展名换成 `.html`，正文写 HTML 即可。

**完整的 HTML 页面**（自带 `<html>`、样式、脚本，例如导出的 notebook 或可视化页面）：在文件最前面加上这段 front matter，页面会原样发布：

```yaml
---
title: 文章标题
layout: none
render_with_liquid: false
---
<!DOCTYPE html>
...
```

文章里的图片放到 `img/` 下，用 `![说明](/img/xxx.png)` 引用。

## 中英双语

网站默认显示英文，右上角的语言按钮会在同一页面内切换中英文。英文版放在中文文件**同一目录**下，文件名加 `_en` 后缀，例如：

```
_posts/2026-05-13-ai-agent-long-horizon-roadmap.md      中文
_posts/2026-05-13-ai-agent-long-horizon-roadmap_en.md   英文
about.md                                                中文
about_en.md                                             英文
```

`_en` 文件不会单独发布成一个页面，只作为对应中文页面的英文版显示。

英文文件只需要 `title`（文章可选 `subtitle`）：

```markdown
---
title: English Title
---

English content...
```

没有英文版的内容在英文模式下会继续显示中文。

完整 HTML 页面（`layout: none`）也可以有英文版，同样命名为 `xxx_en.html`，front matter 和中文版一样带上 `layout: none` 与 `render_with_liquid: false`。英文版会发布在中文地址后面加 `en/`，例如 `/2026/10/11/combo2llmmodel/en/`。构建时会自动在两个版本右下角加上"返回首页"和语言切换按钮，并按访客选的语言自动跳到对应版本，HTML 文件本身不用改。

## 记录想法

在 `_thoughts/` 下新建一个文件，例如 `_thoughts/2026-10-12-something.md`：

```markdown
---
date: 2026-10-12 23:00:00 +0800
en: English version, optional. Markdown works here too.
---
想到什么写什么，支持 Markdown。
```

英文有多段时，`en:` 后面写 `|`，下面每行缩进两个空格（空行可以不缩进）：

```markdown
---
date: 2026-10-12 23:00:00 +0800
en: |
  First paragraph.

  Second paragraph, with a [link](/2026/10/11/combo2llmmodel/en/).
---
中文正文……
```

首页显示最新的 5 条，其余在 `/thoughts/` 页面。

## 本地预览

需要 Ruby 3.x：

```sh
bundle install
bundle exec jekyll serve
```

然后打开 <http://localhost:4000>。
