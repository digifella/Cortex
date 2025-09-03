def display_command_executor_widget(*_args, **_kwargs):
    # No-op stub for Docker minimal distribution
    return None


class SafeCommandExecutor:
    def run(self, *_args, **_kwargs):
        return {"stdout": "", "stderr": "", "returncode": 0}

