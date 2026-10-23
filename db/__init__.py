"""Database package for artwork-automation.

Holds the SQLAlchemy engine/session factory (``db.session``), the declarative
base and ORM models (``db.models``), and small helpers. Only the SERVER
(app.py and src/prompt_registry.py) imports this package; the packaged agent
never does, so no DB driver travels in the agent build.
"""
