# Pairs an English version `name_en.md` with `name.md` in the same folder.
# The English file is not published on its own; templates read it as `page.en`
# (its front matter plus raw `content`, which templates run through markdownify).
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
  end
end
