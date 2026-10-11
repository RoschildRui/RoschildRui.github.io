source "https://rubygems.org"

gem "jekyll", "~> 4.3"
# Pin the Sass converter to the 2.x line so Jekyll uses `sassc` (libsass)
# instead of `sass-embedded`. The latest `sass-embedded` fails to build on CI
# with "uninitialized constant JSON::Fragment"; `sassc` builds reliably.
gem "jekyll-sass-converter", "~> 2.0"
gem "webrick", "~> 1.8"
# No longer a default gem since Ruby 3.4, but still required by Liquid.
gem "bigdecimal"

group :jekyll_plugins do
  gem "jekyll-feed"
end
