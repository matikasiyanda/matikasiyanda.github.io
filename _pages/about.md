---
title: "About"
permalink: /
---

TODO: two or three paragraphs about who you are, what you work on, and what you care about.

## Interests

- TODO
- TODO

## Latest writing

{% for post in site.posts limit:3 %}
- [{{ post.title }}]({{ post.url | relative_url }}) <small>({{ post.date | date: "%-d %b %Y" }})</small>
{% endfor %}

[All posts →](/blog/)
