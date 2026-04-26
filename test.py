import imaplib, os
from worker.intel_mailbox_worker import load_env_file
from pathlib import Path

env = load_env_file(Path("worker/config.env"))
user = env.get("INTEL_IMAP_USERNAME", "")
pw = env.get("INTEL_IMAP_PASSWORD", "")
m = imaplib.IMAP4_SSL("imap.gmail.com", 993)
print("user=", user)
print("pw_len=", len(pw))
print(m.login(user, pw))
m.logout()