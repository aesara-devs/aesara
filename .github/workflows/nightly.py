from datetime import datetime, timezone


suffix = datetime.now(timezone.utc).strftime(r".dev%Y%m%d")
with open("setup.py", "r") as f:
    data = f.read()

data = data.replace(
    "NIGHTLY_VERSION_SUFFIX = None", f'NIGHTLY_VERSION_SUFFIX = "{suffix}"'
)

with open("setup.py", "w") as f:
    f.write(data)
