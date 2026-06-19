"""Drop-in agent plugins.

Every module here self-registers via the ``@register`` decorator and is
auto-imported by ``registry.discover()``. To add a capability, add a file.
To remove one, delete its file. No core file is ever edited.
"""
