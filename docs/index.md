---
layout: default
title: Home
---

Welcome to the open exploration of viable human-AI systems.

## Articles

<ul>
  {% for page in site.pages %}
    {% if page.title and page.url != "/" %}
      <li><a href="{{ site.baseurl }}{{ page.url }}">{{ page.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>
