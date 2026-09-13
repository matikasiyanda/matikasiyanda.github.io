---
title: "Now"
permalink: /now/
---

Problems I'm currently playing with. Updated {{ site.data.problems.updated | date: "%B %Y" }}.

{% for p in site.data.problems.items %}
### {{ p.title }}
{% if p.status %}<span class="badge">{{ p.status }}</span>{% endif %}

{{ p.description | markdownify }}
{% endfor %}
