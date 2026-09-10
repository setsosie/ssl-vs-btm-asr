"""Reporting: turn a directory of runs into the numbers a reader can check.

Three commands sit on this package — ``svb aggregate`` (across-seed mean ± std),
``svb analyze`` (utterance-level intervals and paired tests from the prediction
sidecars) and ``svb data-stats`` (how much audio each language actually
contributes). All three write to ``tables/`` as both Markdown and JSON, so a
number in a write-up can be traced to the file that produced it and regenerated
rather than retyped.
"""
