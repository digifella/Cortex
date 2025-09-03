class _ModelChecker:
    def get_status(self):
        return {"ok": True, "details": "Docker shim"}

model_checker = _ModelChecker()

