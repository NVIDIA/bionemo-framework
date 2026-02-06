# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import hydra
from omegaconf import DictConfig


def _make_html(pdb_text: str) -> str:
    """Make an HTML page with the PDB file."""
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>PDB Viewer</title>
  <script src="https://3dmol.org/build/3Dmol.js"></script>
</head>
<body style="margin:0;">
<div id="viewer" style="width:100vw; height:100vh;"></div>

<script>
  const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
  viewer.addModel(`{pdb_text}`, "pdb");
  viewer.setStyle({{}}, {{ cartoon: {{ color: "spectrum" }} }});
  viewer.zoomTo();
  viewer.render();
</script>
</body>
</html>
"""


@hydra.main(
    config_path="hydra_config",
    config_name="L0_sanity_visualize",
    version_base="1.2",
)
def main(args: DictConfig):
    """Visualize the protein structure using a PDB file. It will serve a simple HTML page with the PDB file.

    Once you run this script, you can view the protein structure in your browser by going to http://{host}:{port} (default http://127.0.0.1:8000)
    """
    pdb_text = Path(args.input_pdb_file).read_text()
    html = _make_html(pdb_text)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        def log_message(self, *args):
            pass  # quiet

    host = "127.0.0.1"  # SSH-safe
    port = args.get("port", 8000)

    print(f"Serving PDB viewer on http://{host}:{port}")
    print("Use SSH port forwarding to view it locally.")

    HTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
