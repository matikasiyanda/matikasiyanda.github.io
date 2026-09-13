---
title: "Blog"
permalink: /blog/
command: "ls -lt ~/blog"
---

<ul class="ls">
{% for post in site.posts %}
  <li><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y-%m-%d" }}</time><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>

Series: [{{ site.posts.first.series }}](/blog/agent-rl/) ·
[glossary](/blog/agent-rl/glossary/) · [references](/blog/agent-rl/references/)
