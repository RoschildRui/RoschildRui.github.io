# Pairs an English version `name_en.md` with `name.md` in the same folder.
# The English file is not published on its own; templates read it as `page.en`
# (its front matter plus raw `content`, which templates run through markdownify).
#
# Standalone HTML pages (`layout: none`) can't switch language in place, so their
# English version is published at `<url>en/`, and both versions get a small
# language bar injected after rendering. Every standalone page also gets the
# site favicon unless it declares its own.
Jekyll::Hooks.register :site, :post_read do |site|
  items = site.pages + site.posts.docs
  by_path = items.to_h { |item| [item.relative_path.sub(/\.[^.\/]+\z/, ""), item] }

  by_path.each do |path, en|
    next unless path.end_with?("_en")

    original = by_path[path.delete_suffix("_en")]
    next unless original

    original.data["en"] = en.data.merge("content" => en.content)
    site.pages.delete(en)
    site.posts.docs.delete(en)
    next unless original.data["layout"] == "none"

    zh_url = original.url
    en_url = "#{zh_url}en/"
    page = Jekyll::PageWithoutAFile.new(site, site.source, en_url, "index.html")
    page.content = en.content
    page.data.merge!(en.data, "permalink" => en_url, "layout" => "none", "zh_url" => zh_url)
    original.data["en_url"] = en_url
    site.pages << page
  end
end

module EnVersions
  def self.bar(site, doc)
    zh = doc.data["en_url"]
    other = "#{site.baseurl}#{zh ? doc.data["en_url"] : doc.data["zh_url"]}"
    home = "#{site.baseurl}/#posts"
    lang = zh ? "zh" : "en"
    head = <<~HTML
      <script>(function(){var l=null;try{l=localStorage.getItem("lang")}catch(e){}if((l==="zh"?"zh":"en")!=="#{lang}")location.replace("#{other}"+location.hash)})();</script>
    HTML
    body = <<~HTML
      <div style="position:fixed;right:16px;bottom:16px;z-index:2147483647;display:flex;gap:6px;font:13px/1 system-ui,-apple-system,sans-serif">
        <a href="#{home}" style="padding:8px 12px;border:1px solid #d0d7de;border-radius:999px;background:#fff;color:#1f2328;text-decoration:none;box-shadow:0 1px 4px rgba(0,0,0,.12)">#{zh ? "← 返回" : "← Home"}</a>
        <a href="#{other}" onclick="try{localStorage.setItem('lang','#{zh ? "en" : "zh"}')}catch(e){}" style="padding:8px 12px;border:1px solid #d0d7de;border-radius:999px;background:#fff;color:#1f2328;text-decoration:none;box-shadow:0 1px 4px rgba(0,0,0,.12)">#{zh ? "EN" : "中"}</a>
      </div>
    HTML
    [head, body]
  end

  def self.icons(site)
    base = site.baseurl
    <<~HTML
      <link rel="icon" href="#{base}/img/favicon.ico" sizes="any">
      <link rel="icon" type="image/png" href="#{base}/img/favicon-32.png">
      <link rel="apple-touch-icon" href="#{base}/img/apple-touch-icon.png">
    HTML
  end

  def self.inject(doc)
    return unless doc.data["layout"] == "none" && doc.output_ext == ".html"

    head = doc.output.match?(/<link[^>]+rel=["'][^"']*icon/i) ? "" : icons(doc.site)
    if doc.data["en_url"] || doc.data["zh_url"]
      script, body = bar(doc.site, doc)
      head = script + head
      doc.output = doc.output.sub(%r{</body>}i) { "#{body}</body>" }
    end
    doc.output = doc.output.sub(/<head[^>]*>/i) { |tag| "#{tag}\n#{head}" } unless head.empty?
  end
end

Jekyll::Hooks.register [:pages, :documents], :post_render do |doc|
  EnVersions.inject(doc)
end
