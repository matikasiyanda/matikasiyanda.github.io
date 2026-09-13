---
title: "Writing"
permalink: /blog/
---

<ul class="list">
{% for post in site.posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%-d %b %Y" }}</time></li>
{% endfor %}
</ul>

Series: [{{ site.posts.first.series }}](/blog/agent-rl/) ·
[glossary](/blog/agent-rl/glossary/) · [references](/blog/agent-rl/references/)
