class _NoopMigrationManager:
    def __init__(self):
        pass
    def migrate(self, *_args, **_kwargs):
        return None


def create_migration_manager():
    """Return a no-op migration manager in Docker minimal build.
    The full migration is only needed when converting legacy Docling formats.
    """
    return _NoopMigrationManager()

