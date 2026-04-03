import os

from openreward.environments import Server

from miroeval import MiroEval

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    server = Server([MiroEval])
    server.run(port=port)
