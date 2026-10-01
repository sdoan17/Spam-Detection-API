import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    repo_id = os.environ["HF_REPO_ID"].strip()
    revision = os.environ["HF_REVISION"].strip()
    token = os.environ.get("HF_TOKEN") or False

    if not repo_id or not revision:
        raise ValueError("Set HF_REPO_ID and HF_REVISION.")

    project_dir = Path(__file__).resolve().parents[1]
    model_dir = project_dir / "models"

    filenames = [
        "spam-model.keras",
        "spam-classifier-tokenizer.json",
        "spam-classifier-metadata.json",
    ]

    # Resolve the version once so every file comes from the same commit.
    api = HfApi(token=token)
    resolved_revision = api.model_info(
        repo_id=repo_id,
        revision=revision,
    ).sha

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=resolved_revision,
        allow_patterns=filenames,
        local_dir=model_dir,
        token=token,
    )

    for filename in filenames:
        path = model_dir / filename

        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty model file: {filename}")

        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))

        print(f"Downloaded: {filename}")

    # Record exactly which model version was downloaded.
    manifest = {
        "repo_id": repo_id,
        "revision": resolved_revision,
        "files": filenames,
    }

    (model_dir / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Model revision: {resolved_revision}")


if __name__ == "__main__":
    main()