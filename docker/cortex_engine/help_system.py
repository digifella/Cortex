class _HelpSystem:
    def show_contextual_help(self, *_args, **_kwargs):
        return None

    def show_help_modal(self, *_args, **_kwargs):
        return None

    def show_help_menu(self, *_args, **_kwargs):
        # Minimal no-op for Docker distribution
        return None

help_system = _HelpSystem()
