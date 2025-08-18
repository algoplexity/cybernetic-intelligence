---
layout: default
title: Home
---

# Cybernetic Intelligence

Welcome to the open exploration of viable human-AI systems.

## 📄 Articles

<ul>
  {% assign pages = site.pages | sort: 'title' %}
  {% for p in pages %}
    {% if p.title and p.url != "/" %}
      <li>
        <a href="{{ p.url | relative_url }}">{{ p.title }}</a>
      </li>
    {% endif %}
  {% endfor %}
</ul>
