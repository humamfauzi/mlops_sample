#!/usr/bin/env python3
import os
import sys

import uvicorn

import buildinfo
from server.main import app

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Read port from environment variable, default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    prov = buildinfo.describe()
    print(
        f"mlops server starting: version={prov['version']} "
        f"git_sha={prov['git_sha']} build_time={prov['build_time']}"
    )
    print(f"listening on http://{host}:{port}")

    uvicorn.run(app, host=host, port=port)
